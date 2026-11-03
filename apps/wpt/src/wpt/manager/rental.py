"""Renting gpu workers (docs/machines.md, "Rented gpu workers"): one small
interface of ours, which each provider the repository integrates
implements, and the settings it reads.

    Provider     rent(name, token, interruptible, count), address(name), addresses(name, count),
                 stop(name); cluster: whether count machines can be rented in one request
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

    def address(self, name: str) -> Address:
        """Where the worker of the one-machine rental name is, once it
        listens. RentalError when it does not within WPT_RENTAL_WAIT."""
        raise NotImplementedError

    def addresses(self, name: str, count: int) -> list[Address]:
        """Where the workers of the cluster name are, in order, once every
        one listens; each address gives the cluster as its place and the
        machine's private address there. RentalError when they do not
        within WPT_RENTAL_WAIT."""
        raise NotImplementedError

    def stop(self, name: str):
        """Stop the rental name and give its machines back; nothing when the
        provider has no such rental (any more). RentalError when it
        cannot."""
        raise NotImplementedError


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
