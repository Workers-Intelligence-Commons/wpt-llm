"""Writing tokens through the link (docs/machines.md): the model lives on a
gpu worker, and what asks it to write (an environment's part on the wpt
manager, a test of wpt-eval) lives on the manager, which holds the
tokenizer.

    Writers          the writers (wpt_inference.Writer) of one model on this machine, started and
                     written with by number
    Writing          Writers on a job's link: the requests writer.start and writer.write
    RemoteWriter     the manager's side of one such writer: the writer an attempt's loop
                     writes with, whose tokens come from the gpu worker's model
                     (worker_generator() asks for a seeded generator)
    LocalWriter      the same writer on Writers of this process, for what runs beside the model

A request writes as much as one round trip can: writer.write goes on until
each row asked has written a stop token or reached its limit. A caller that wants a
seeded generator asks for one (generator.new: a seed in, a number out) and
names that number in writer.start: the writers that name it draw from
it, each going on from where the last one left it, as the one generator a
caller passes to wpt_inference.Writer does. A writer that names none draws from
the generator the job gave Writing.
"""

import contextlib

import torch


class Writers:
    """The writers of model, each by the number start() gave it. Starting a
    writer ends the ones before it: their caches go, and what they recorded
    of the routes stays (routes()). generator: the one a writer that
    names none draws from."""

    def __init__(self, model, generator=None):
        self.model, self.generator = model, generator
        self.writers, self.routes_of, self.generators = {}, {}, {}

    def new_generator(self, seed: int) -> int:
        """A new torch.Generator on the model's device, seeded with seed; its number."""
        number = len(self.generators) + 1
        device = next(self.model.parameters()).device
        self.generators[number] = torch.Generator(device=device).manual_seed(seed)
        return number

    def start(self, prompt, rows, *, temperature, top_k, stops, routes=False, generator=None) -> int:
        from wpt_inference import Writer

        for number, writer in self.writers.items():  # a new writer: the earlier ones never write again
            self.routes_of[number] = writer.routes
        self.writers = {}
        number = len(self.routes_of) + 1
        self.writers[number] = Writer(self.model, prompt, rows, temperature=temperature, top_k=top_k,
                                      generator=self.generators[generator] if generator else self.generator, stops=stops,
                                      routes=routes)
        return number

    def write(self, number, inputs, limits) -> dict:
        """{row: the tokens it wrote}, each row going on until it wrote a
        stop token or reached its limit (inputs and limits as
        wpt_inference.Writer.write takes them)."""
        writer = self.writers[number]
        written, left = {row: [] for row in limits}, dict(limits)
        with torch.no_grad():
            while left:
                for row, tokens in writer.write(inputs, left).items():
                    written[row] += tokens
                    if tokens[-1] in writer.stops or len(written[row]) >= limits[row]:
                        del left[row]
                    else:
                        left[row] = limits[row] - len(written[row])
                inputs = {}
        return written

    def routes(self, number, row):
        """The recorded routes of one row of writer number (wpt_model.Routes)."""
        found = self.routes_of.get(number)
        if found is None:
            found = self.writers[number].routes
        return found[row]


class Writing:
    """Writers on a job's link: while active (called with the model and the
    generator its writers draw from, when they have no seed), the manager's
    requests writer.start and writer.write are answered. routes(handle)
    gives a row's recorded routes back: the handle is what the manager's
    writer gave for it."""

    KINDS = ("writer.start", "writer.write", "generator.new")

    def __init__(self, link):
        self.link, self.writers = link, None

    @contextlib.contextmanager
    def __call__(self, model, generator=None):
        self.writers = Writers(model, generator)
        self.link.handlers.update({"writer.start": self._start, "writer.write": self._write,
                                   "generator.new": self._generator})
        try:
            yield self
        finally:
            for kind in self.KINDS:
                self.link.handlers.pop(kind, None)
            self.writers = None

    def routes(self, handle):
        return self.writers.routes(handle["writer"], handle["row"])

    def _start(self, header, body):
        number = self.writers.start(header["prompt"], header["rows"], temperature=header["temperature"],
                                    top_k=header["top_k"], stops=header["stops"], routes=header["routes"],
                                    generator=header.get("generator"))
        return {"writer": number}, b""

    def _generator(self, header, body):
        return {"generator": self.writers.new_generator(header["seed"])}, b""

    def _write(self, header, body):
        inputs = {int(row): tokens for row, tokens in header["inputs"].items()}
        limits = {int(row): limit for row, limit in header["limits"].items()}
        written = self.writers.write(header["writer"], inputs, limits)
        return {"written": {str(row): tokens for row, tokens in written.items()}}, b""


def worker_generator(worker, channel, seed) -> int:
    """A new generator on the gpu worker's model, seeded with seed: the
    number its writers name."""
    header, _ = worker.call(channel, "generator.new", seed=seed)
    return header["generator"]


class RemoteWriter:
    """wpt_inference.Writer's write and routes, for a writer started on the
    gpu worker's model (writer.start, through the job's channel) and
    written with (writer.write), as the loop of an attempt writes with
    one. routes: each row's handle, which the worker's part turns back into
    the row's routes. generator: a number worker_generator() gave."""

    def __init__(self, worker, channel, prompt, rows=1, *, temperature, top_k, stops, routes=False, generator=None):
        self.worker, self.channel, self.rows, self.recorded = worker, channel, rows, routes
        self.stops = set(stops)
        header, _ = worker.call(channel, "writer.start", prompt=list(prompt), rows=rows, temperature=temperature,
                                top_k=top_k, stops=list(stops), routes=routes, generator=generator)
        self.number = header["writer"]

    @property
    def routes(self):
        return [{"writer": self.number, "row": row} for row in range(self.rows)] if self.recorded else None

    def write(self, inputs, limits):
        header, _ = self.worker.call(self.channel, "writer.write", writer=self.number,
                                     inputs={str(row): list(tokens) for row, tokens in inputs.items()},
                                     limits={str(row): limit for row, limit in limits.items()})
        return {int(row): tokens for row, tokens in header["written"].items()}


class LocalWriter:
    """RemoteWriter's calls on Writers of this process."""

    def __init__(self, writers, prompt, rows=1, *, temperature, top_k, stops, routes=False, generator=None):
        self.writers, self.stops = writers, set(stops)
        self.number = writers.start(list(prompt), rows, temperature=temperature, top_k=top_k, stops=list(stops),
                                    routes=routes, generator=generator)

    @property
    def routes(self):
        return self.writers.writers[self.number].routes

    def write(self, inputs, limits):
        return self.writers.write(self.number, inputs, limits)
