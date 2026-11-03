"""Renting gpu workers (docs/machines.md, "Rented gpu workers"): one small
interface of ours, which each provider the repository integrates
implements, and the settings it reads.

    Provider     rent(name, token, interruptible, count), address(name, watch),
                 addresses(name, count, watch), stop(name), price(interruptible, count);
                 cluster: whether count machines can be
                 rented in one request
    PROVIDERS    {setting's value: the provider's class}: SkyPilot (skypilot.py), which reaches
                 the clouds it supports; code of our own for a provider SkyPilot lacks or
                 handles badly is one more class here

The manager names each rental before it rents it, and keeps that name on
its own disk first (machines.py), so a rental a crash interrupted is still
one it can stop. It sets each rental's token itself (WPT_WORKER_TOKEN of
its workers), which is never printed whole.

Settings, from the environment, else from the .env file at the
repository's root (provider keys come the same way, as each provider
reads them):

    WPT_RENTAL_PROVIDER  who rents the gpu workers (default skypilot)
    WPT_RENTAL_KIND      on-demand (the default), or interruptible (spot: cheaper, and taken back
                         when the provider needs it; a worker taken back is a lost worker)
    WPT_RENTAL_GPUS      what each rented machine has, in the provider's own words (SkyPilot:
                         H100:8); required to rent
    WPT_RENTAL_WAIT      seconds to wait for a rented worker to give its address (default 1800)
    WPT_BUDGET           the most a run may spend on rentals, in dollars, at the provider's price
    WPT_MAX_HOURS        the most hours a run may hold rentals; with both, the first reached stops
                         the run (docs/machines.md, "The cost cap")
"""

from dataclasses import dataclass

ON_DEMAND, INTERRUPTIBLE = "on-demand", "interruptible"
KINDS = (ON_DEMAND, INTERRUPTIBLE)
WAIT = 1800.0


class RentalError(Exception):
    """A provider could not rent, reach or stop a gpu worker, or its
    settings are not usable."""


@dataclass(frozen=True)
class Address:
    """Where a rented gpu worker is: what the manager reaches it at (host,
    port), the sha256 of its certificate (pin), and, when the provider
    knows, the place it is in and its private address there."""
    host: str
    port: int
    pin: str
    place: str | None = None
    private: str | None = None


@dataclass(frozen=True)
class Cap:
    """What a run may spend on rentals, over all its starts (None: no such
    cap)."""
    dollars: float | None = None
    hours: float | None = None

    def reached(self, dollars: float, hours: float) -> str | None:
        """What says the run must stop, having spent that much; None when
        neither cap is reached."""
        if self.dollars is not None and dollars >= self.dollars:
            return f"the cost cap: ${dollars:.2f} spent, WPT_BUDGET is ${self.dollars:.2f}"
        if self.hours is not None and hours >= self.hours:
            return f"the cost cap: {hours:.2f} hours rented, WPT_MAX_HOURS is {self.hours:.2f}"
        return None

    def projected(self, rate: float, dollars: float, hours: float) -> str:
        """What a run that spent that much comes to at rate dollars an
        hour, for people to read before it rents."""
        left = [(self.hours - hours) if self.hours is not None else None,
                (self.dollars - dollars) / rate if self.dollars is not None and rate else None]
        left = [max(hours, 0.0) for hours in left if hours is not None]
        if not left:
            return f"${rate:.2f} an hour, ${dollars:.2f} spent so far; no cap (WPT_BUDGET, WPT_MAX_HOURS) limits the run"
        return (f"${rate:.2f} an hour, ${dollars:.2f} spent so far; the cost cap stops the run after {min(left):.2f} "
                f"more hours, at ${dollars + min(left) * rate:.2f} in all")


class Provider:
    """One way of renting gpu workers. Each machine runs one gpu worker
    (deploy/gpu-worker/start.sh) whose token is the one rent() gives, and
    stops once that worker ends (a worker with no manager ends after
    WPT_WORKER_WAIT). settings: the manager's settings; root: the
    repository's root. A provider that can rent several machines in one
    request, as one cluster that reaches itself on a private network,
    says so with cluster = True."""

    cluster = False

    def __init__(self, settings: dict, root):
        self.settings, self.root = settings, root

    def rent(self, name: str, token: str, interruptible: bool, count: int = 1):
        """Rent count machines under name, each running a gpu worker with
        token; with count above 1, as one cluster (only a provider with
        cluster = True is asked to). Returns once the provider has started
        them. RentalError when it cannot."""
        raise NotImplementedError

    def address(self, name: str, watch) -> Address:
        """Where the worker of the one-machine rental name is, once it
        listens. RentalError when it does not within WPT_RENTAL_WAIT. It
        calls watch() again and again as it waits: what that raises ends the
        wait."""
        raise NotImplementedError

    def addresses(self, name: str, count: int, watch) -> list[Address]:
        """Where the workers of the cluster name are, in order, once every
        one listens; each address gives the cluster as its place and the
        machine's private address there. RentalError when they do not
        within WPT_RENTAL_WAIT. watch: as in address()."""
        raise NotImplementedError

    def stop(self, name: str):
        """Stop the rental name and give its machines back; nothing when the
        provider has no such rental (any more). RentalError when it
        cannot."""
        raise NotImplementedError

    def price(self, interruptible: bool, count: int = 1) -> float:
        """What count machines, as rent() would rent them, cost in dollars
        an hour, all of them together. RentalError when the provider does
        not say."""
        raise RentalError(f"{type(self).__name__} gives no price")


def providers() -> dict:
    from .skypilot import SkyPilot

    return {"skypilot": SkyPilot}


def provider(settings, root) -> Provider:
    """The provider the settings name. RentalError for an unknown one."""
    name = settings.get("WPT_RENTAL_PROVIDER") or "skypilot"
    known = providers()
    if name not in known:
        raise RentalError(f"WPT_RENTAL_PROVIDER={name!r}: one of {', '.join(sorted(known))}")
    return known[name](settings, root)


def kind(settings) -> str:
    found = settings.get("WPT_RENTAL_KIND") or ON_DEMAND
    if found not in KINDS:
        raise RentalError(f"WPT_RENTAL_KIND={found!r}: one of {', '.join(KINDS)}")
    return found


def wait(settings) -> float:
    try:
        return float(settings.get("WPT_RENTAL_WAIT") or WAIT)
    except ValueError:
        raise RentalError(f"WPT_RENTAL_WAIT={settings['WPT_RENTAL_WAIT']!r}: a number of seconds") from None


def cap(settings) -> Cap:
    """The cap the settings give. RentalError for one that is not a number
    above 0."""
    found = {}
    for field, setting in (("dollars", "WPT_BUDGET"), ("hours", "WPT_MAX_HOURS")):
        text = settings.get(setting)
        if text:
            try:
                found[field] = float(text)
            except ValueError:
                found[field] = 0.0
            if not 0 < found[field] < float("inf"):
                raise RentalError(f"{setting}={text!r}: a number above 0")
    return Cap(**found)
