"""The record of one WPT command run: what it read, what it did and what it
wrote, so that afterwards one can know exactly what happened.

wpt-dataset build, wpt-train and wpt-eval each open one
Telemetry per run. Its record is its own folder of the store named by
WPT_STORE, records/<folder>/ (see wpt_store), named
<UTC start>-<command>-<main id>-<8 random hex digits>, where the main id is
what the run is about (a recipe name, a short dataset id, a short weights
hash: the command chooses):

    start.json     the command, its arguments, when it started, the tools,
                   the code (its commit and a hash of its uncommitted
                   changes), the repository commit, and the trace id the live backend
                   shows it under
    events.jsonl   one JSON object per line, in the order they happened:
                   spans opened and closed, events with their values, and
                   every line the command printed
    ids.json       the ids the run read (inputs) and wrote (outputs), such
                   as a dataset id or a weight file's sha256, rewritten as
                   each is learned: the chain from a dataset build to the
                   models trained and evaluated on it
    end.json       how it ended: succeeded or failed, the exit code, the
                   error and its traceback
    other files    written by a part of the command (write_file, append_file)

Each file written whole is in the store before the call that makes it
returns; a run with no end.json did not reach its end. The store holds
added lines back (see wpt_store): they are sent every FLUSH_SECONDS, at
the end of every span, before every step that cannot be undone
(irreversible) and at the end of the run, so a crash loses at most the
lines of the last few seconds. When the store does not take a part,
RecordError is raised, and every later call raises it again without
writing anything: the command stops, since without its record it must do
nothing more (wpt-train's training process still saves its checkpoint
first, unrecorded), and exits with RECORD_FAILED. A send that fails in the
background interrupts the main thread, as Ctrl+C would: on Linux and macOS
the flush thread sends the main thread a real SIGINT, which also cuts short
a wait (time.sleep, waiting for a process, reading a socket), and Python
acts on it as soon as the main thread runs Python code again (a long
computation in a library, such as a GPU step, ends first); on Windows,
_thread.interrupt_main, which takes effect only once the main thread runs
Python code again, after a wait ends. The KeyboardInterrupt it raises
leaves the with block, and close() turns it into RecordError, never into a
stop by the user (exit code 130). Where the main thread runs a handler of
its own for Ctrl+C (wpt-train's training loop, its launcher), that handler
runs instead and the next call raises RecordError (wpt-train's training
loop first finishes its step and saves its checkpoint). In a process that
ignores Ctrl+C (started in the background by a shell that is not
interactive), or leaves it to the system's default, there is no
interruption and the next call raises RecordError. A record that broke
always ends in RecordError when the with block ends, even when the block
itself ended without an error.

A file name that the store does not allow, or a file both written whole
and added to, is a mistake of the caller: ValueError or the store's
MixedRecord, raised as itself, and the record goes on.

After close(), every recording call raises RuntimeError: the run is over,
and a call then is a mistake in the command's code.

When OTEL_EXPORTER_OTLP_ENDPOINT is set (environment, or the .env file the
command reads WPT_STORE from), the same spans, the events and printed lines
(as log records) and the events' values (as gauges) also go to that
OpenTelemetry backend over OTLP/HTTP, in the background. A backend that
fails gets one warning on stderr; the run goes on and each later batch is
tried again quietly, so the live view resumes when the backend is back.
The store record is the account; the live view is a convenience.

A process started by another one for the same run (wpt-train's launcher
starts the training processes) joins its record: the launcher writes
start.json and end.json, the joined process its own spans, events and ids.
"""

import _thread
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import secrets
import signal
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags

from wpt_store import MissingObject, MixedRecord, as_record_name, check_record_name, read_env_file
from wpt_utils.files import sha256_file
from wpt_utils.programs import code, repository

RECORD_FAILED = 3  # the exit code of a command whose record could not be written; never restarted
ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
JOIN_FOLDER = "WPT_RECORD_FOLDER"  # set by a launcher for the processes that join its record
JOIN_PARENT = "WPT_RECORD_PARENT"  # <trace id>-<span id> of the launcher's span they belong under
START, EVENTS, IDS, END = "start.json", "events.jsonl", "ids.json", "end.json"
FLUSH_SECONDS = 5  # the store is sent the lines it holds back at least this often
FLUSH_THREAD = "wpt-telemetry-flush"
SHORT = 8  # characters of a hash or a dataset id in a record folder's name
METRIC_INTERVAL_MS = 10_000  # live values are sent every 10 s unless OTEL_METRIC_EXPORT_INTERVAL says otherwise
PACKAGES = ("torch",)  # the outside packages whose version start.json names; the code itself is code()


class RecordError(Exception):
    """The store did not take a part of the record; nothing more is written
    and the command must stop. The record keeps every part written before."""


class Span:
    """An open span: set() adds attributes, recorded when it closes."""

    def __init__(self, id, otel, name=None):
        self.id, self.attributes, self.otel, self.name = id, {}, otel, name
        self.started = time.monotonic()

    def set(self, **attributes):
        self.attributes.update(attributes)


class Telemetry:
    """One command run's record; see the module's docstring. Built by
    start(), join() or off(); a with block closes it and records how the
    run ended."""

    def __init__(self, store, folder, command, live, echo, owner, parent=None):
        self.store, self.folder, self.command = store, folder, command
        self._live, self._echo, self._owner = live, echo, owner
        # re-entrant: an interruption (the flush thread's, or Ctrl+C) can land just after the main
        # thread took it and leave it taken; the main thread can then still take it again in close()
        self._lock = threading.RLock()
        self._broken = None  # why the store refused a part; then everything is refused
        self._started, self._clock = time.time(), time.monotonic()
        self._closed = False
        self._closing = False  # close() has begun: the flush thread no longer interrupts (under _lock)
        self._root = None  # the run's own span
        self._threads = threading.local()  # each thread's stack of open spans (_stack)
        if live is not None:
            root = live.start(command, parent, {})
            self._root = Span(_span_id(root), root, command)
            self.trace = f"{root.get_span_context().trace_id:032x}"
        self._stop = threading.Event()
        self._flusher = None  # started by start() and join() once their first parts are in the store

    # -- making one ----------------------------------------------------------------

    @classmethod
    def start(cls, store, command, arguments, main_id, inputs=None, env_file=".env", environ=None, echo=None):
        """A new run's record, in a new folder named after main_id (what the
        run is about: a recipe name, short(dataset id)...): start.json and
        ids.json are written before it returns (RecordError when the store
        refuses them)."""
        moment = datetime.now(UTC)
        folder = f"{moment:%Y%m%dT%H%M%SZ}-{command}-{as_record_name(main_id)}-{secrets.token_hex(4)}"
        live = _Live(command, folder, _endpoint(env_file, environ), echo or _print)
        telemetry = cls(store, folder, command, live, echo or _print, owner=True)
        try:
            telemetry._write(START, _json({"command": command, "arguments": list(arguments),
                                           "started": _iso(moment), "trace": telemetry.trace,
                                           "inputs": dict(inputs or {}), **_where()}))
            telemetry._write(IDS, _json({"inputs": dict(inputs or {}), "outputs": {}}))
            telemetry._line({"kind": "span_start", "span": telemetry._root.id, "parent": None, "name": command,
                             "attributes": {}})
        except RecordError:
            telemetry._abandon()
            raise
        telemetry._start_flushing()
        return telemetry

    @classmethod
    def join(cls, store, command, environ, env_file=".env", echo=None):
        """The record a launcher opened for this run (environ[JOIN_FOLDER]),
        under its span; this process writes events and ids, not start.json
        nor end.json."""
        folder = environ[JOIN_FOLDER]
        trace_id, span_id = (int(part, 16) for part in environ[JOIN_PARENT].split("-"))
        parent = SpanContext(trace_id, span_id, is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED))
        try:
            store.read_record(folder, START)
        except Exception as error:
            raise RecordError(f"the record {folder} of this run could not be read from {store}: "
                              f"{type(error).__name__}: {error}") from None
        live = _Live(command, folder, _endpoint(env_file, environ), echo or _print)
        telemetry = cls(store, folder, command, live, echo or _print, owner=False, parent=parent)
        try:
            telemetry._line({"kind": "span_start", "span": telemetry._root.id, "parent": f"{span_id:016x}",
                             "name": command, "attributes": {"process": "joined"}})
        except RecordError:
            telemetry._abandon()
            raise
        telemetry._start_flushing()
        return telemetry

    @classmethod
    def off(cls, echo=None):
        """Records nothing; log() still prints, through echo (default:
        stdout, stderr for errors). For a process whose events the main
        one records, and for code run outside a command."""
        return cls(None, None, "", None, echo or _print, owner=False)

    def child_environment(self) -> dict:
        """What a process started for this run needs to join its record,
        under this thread's current span, and to send to the same live
        backend. The lines held back are sent first, so the record is whole
        up to here."""
        self._check_open()
        if self.store is None:
            return {}
        self._flush()
        endpoint = self._live.failure.endpoint if self._live and self._live.failure else None
        return {JOIN_FOLDER: self.folder, JOIN_PARENT: f"{self.trace}-{self._stack[-1].id}",
                **({ENDPOINT: endpoint} if endpoint else {})}

    # -- recording -----------------------------------------------------------------

    # the names before / are positional only, so any attribute name is free

    def log(self, text, /, error=False, **attributes):
        """Record a printed line, then print it (stderr when error)."""
        self._check_open()
        self._line({"kind": "log", "span": self._span(), "severity": "error" if error else "info", "text": str(text),
                    **({"attributes": attributes} if attributes else {})})
        if self._live:
            self._live.emit(self._top(), None, str(text), attributes, error)
        self._echo(str(text), error)

    def event(self, name, /, values=None, **attributes):
        """Record that something happened: attributes say what (anything
        JSON holds), values are its numbers, also sent live as gauges named
        wpt.<name>.<value name>."""
        self._check_open()
        self._line({"kind": "event", "span": self._span(), "name": name, "attributes": attributes,
                    **({"values": values} if values else {})})
        if self._live:
            self._live.emit(self._top(), name, None, {**attributes, **(values or {})}, False)
            for key, value in (values or {}).items():
                self._live.gauge(f"wpt.{name}.{key}", value)

    def span(self, name, /, **attributes):
        """A part of the run, from the with block's start to its end; spans
        and events inside it are recorded under it. An exception leaving it
        is recorded as its error, with its traceback."""
        return self.finish(self.begin(name, **attributes))

    # Each thread has its own open spans: what a thread records goes under
    # the span it opened last, or the run's own span in a thread that opened
    # none. The four methods below let work handed to other threads record
    # under a span of its own: begin() opens one in one thread, within()
    # makes it the place a thread records in, end() or finish() ends it,
    # from any thread.

    def current(self):
        """The span this thread records under now (None when the record
        has no span: off())."""
        return self._stack[-1] if self._stack else None

    def begin(self, name, /, **attributes):
        """Open a span under this thread's current one, without recording
        anything under it yet (within, finish); it stays open until end()
        or finish() ends it."""
        self._check_open()
        if self.store is None and self._live is None:
            return Span(None, None, name)
        live = self._live.start(name, self._top(), attributes)
        span = Span(_span_id(live), live, name)
        self._line({"kind": "span_start", "span": span.id, "parent": self._span(), "name": name,
                    "attributes": attributes})
        return span

    @contextlib.contextmanager
    def within(self, span):
        """Record under span (current() of another thread, or begin()'s) in
        this thread for the with block's length; span stays open."""
        self._stack.append(span)
        try:
            yield span
        finally:
            self._stack.pop()

    @contextlib.contextmanager
    def finish(self, span):
        """within(span), then end span when the with block ends, with the
        exception leaving it as its error."""
        try:
            with self.within(span):
                yield span
        except RecordError:
            raise
        except BaseException as error:
            self.end(span, error)
            raise
        self.end(span)

    def end(self, span, error=None):
        """End span (begin()'s), with error as its error when given."""
        self._check_open()
        if span.id is None and span.otel is None:
            return
        self._end_span(span, span.name, span.started, error)

    def input(self, **ids):
        """Ids this run read (dataset id, weights' sha256, recipe name...)."""
        self._check_open()
        self._add_ids("inputs", "input", ids)

    def output(self, **ids):
        """Ids this run wrote."""
        self._check_open()
        self._add_ids("outputs", "output", ids)

    def write_file(self, name, data: bytes):
        """Write a file of this run into its record folder, whole."""
        self._check_open()
        self._check_name(name)
        self._write(name, data)
        self._line({"kind": "file", "span": self._span(), "name": name, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()})

    def append_file(self, name, data: bytes):
        """Add lines to a file of this run in its record folder (such as one
        JSON line per item), durable as the record's own lines are."""
        self._check_open()
        self._check_name(name)
        self._append(name, data)

    def irreversible(self, name, /, values=None, **attributes):
        """Record, as event() does, a step that cannot be undone, just before
        taking it, and send it with every line the store holds back: the
        step happens only once its record is in the store. RecordError when
        the store refuses it, and the caller must not take the step."""
        self.event(name, values, **attributes)
        self._flush()

    # -- ending --------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.close(error)
        return False

    def close(self, error=None):
        """Record how the run ended (error: what stopped it, if anything),
        then send what is left to the live backend. When the record broke,
        meanwhile or before (a send in the background), nothing is written
        and RecordError is raised, from error: the command must exit with
        RECORD_FAILED, whatever else stopped it."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stop_flushing()
            try:
                if self.store is not None and self._broken is None and not isinstance(error, RecordError):
                    status, code, text = _ending(error)
                    self._end_span(self._root, self.command, self._clock, error if status != "succeeded" else None)
                    if self._owner:
                        self._write(END, _json({"status": status, "exit_code": code,
                                                "error": text,
                                                "traceback": _traceback(error) if status != "succeeded" else None,
                                                "ended": _iso(datetime.now(UTC)),
                                                "seconds": round(time.time() - self._started, 3)}))
                        self._end_records()
            finally:
                if self._live:
                    self._live.close()
        except KeyboardInterrupt:  # the flush thread's interruption, or a Ctrl+C, arriving only now
            if self._broken is None:
                raise
            if self._live:
                self._live.close()  # what the interruption cut short
        if self._broken is not None and not isinstance(error, RecordError):
            raise RecordError(self._broken) from error

    # -- inside --------------------------------------------------------------------

    @property
    def _stack(self):
        """This thread's open spans, the run's own first."""
        stack = getattr(self._threads, "stack", None)
        if stack is None:
            stack = self._threads.stack = [self._root] if self._root is not None else []
        return stack

    def _span(self):
        return self._stack[-1].id if self._stack else None

    def _top(self):
        return self._stack[-1].otel if self._stack else None

    def _end_span(self, span, name, started, error):
        line = {"kind": "span_end", "span": span.id, "name": name, "status": "error" if error is not None else "ok",
                "seconds": round(time.monotonic() - started, 6), "attributes": span.attributes}
        if error is not None:
            line["error"] = _describe(error)
            line["traceback"] = _traceback(error)
        self._line(line)
        self._flush()
        if self._live:
            self._live.end(span.otel, span.attributes, line.get("error"))

    def _add_ids(self, part, kind, ids):
        """Add ids to ids.json as the store holds it now, so that a process
        never writes back an older copy than another process left."""
        self._line({"kind": kind, "span": self._span(), "ids": ids})

        def merge():
            try:
                known = json.loads(self.store.read_record(self.folder, IDS))
            except MissingObject:
                known = {"inputs": {}, "outputs": {}}
            known[part].update(ids)
            self.store.write_record(self.folder, IDS, _json(known))

        self._store_call(IDS, merge)

    def _check_name(self, name):
        """ValueError for a name the record keeps for itself or the store does
        not allow, before anything reaches the store."""
        if name in (START, EVENTS, IDS, END):
            raise ValueError(f"{name} is the record's own file; give another name")
        check_record_name(name)

    def _line(self, line):
        if self.store is None:
            return
        self._append(EVENTS, (json.dumps({"time": _iso(datetime.now(UTC)), **line}, ensure_ascii=False,
                                         default=str) + "\n").encode())

    def _append(self, name, data):
        self._store_call(name, lambda: self.store.append_record(self.folder, name, data))

    def _write(self, name, data):
        self._store_call(name, lambda: self.store.write_record(self.folder, name, data))

    def _end_records(self):
        """The record ended: a local store compresses the file it added the
        events to into one part."""
        if self.store is not None:
            self._store_call(EVENTS, lambda: self.store.end_records(self.folder))

    def _flush(self):
        if self.store is not None:
            self._store_call(EVENTS, self.store.flush_records)

    def _start_flushing(self):
        """Send the lines a store holds back every FLUSH_SECONDS, from now on:
        only once the record's first parts are written, so that an
        interruption never reaches start() or join() before they return."""
        self._flusher = threading.Thread(target=self._flush_regularly, name=FLUSH_THREAD, daemon=True)
        self._flusher.start()

    def _flush_regularly(self):
        while not self._stop.wait(FLUSH_SECONDS):
            try:
                self._flush()
            except RecordError:
                with self._lock:
                    if self._closing:
                        return  # close() raises it
                _interrupt_main()  # the command stops now, not at its next call; see the module's docstring
                return

    def _stop_flushing(self):
        """Stop the flush thread; once this returns, it interrupts nothing."""
        with self._lock:
            self._closing = True
        self._stop.set()
        if self._flusher is not None and self._flusher is not threading.current_thread():
            self._flusher.join()

    def _abandon(self):
        """Give up a record whose start the store refused."""
        self._closed = True
        self._stop_flushing()
        if self._live:
            self._live.close()

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"the record {self.folder} of {self.command} is closed: nothing more can be recorded")

    def _store_call(self, name, call):
        if self.store is None:
            return
        with self._lock:
            if self._broken is not None:
                raise RecordError(self._broken)
            try:
                call()
            except MixedRecord:  # the caller's mistake, refused before anything was written: the record goes on
                raise
            except Exception as error:  # whatever the folder or the bucket raised: the part is not stored
                self._broken = (f"the record of this run could not be written to {self.store} "
                                f"(records/{self.folder}/{name}): {type(error).__name__}: {error}")
                raise RecordError(self._broken) from error


class _Live:
    """The OpenTelemetry side: spans, log records and gauges, sent to the
    backend at endpoint when there is one. Without one, it only gives the
    spans their ids."""

    def __init__(self, command, folder, endpoint, echo):
        resource = Resource.create({"service.name": command, "service.instance.id": folder, "wpt.record": folder})
        self.traces = TracerProvider(resource=resource)
        self.logs = LoggerProvider(resource=resource)
        readers = []
        self.failure = None
        if endpoint:
            _quiet_exporters()
            from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            failure = self.failure = _Failure(endpoint, echo)
            base = endpoint.rstrip("/")
            self.traces.add_span_processor(BatchSpanProcessor(_Spans(OTLPSpanExporter(f"{base}/v1/traces"), failure)))
            self.logs.add_log_record_processor(
                BatchLogRecordProcessor(_Logs(OTLPLogExporter(f"{base}/v1/logs"), failure)))
            readers.append(PeriodicExportingMetricReader(_Metrics(OTLPMetricExporter(f"{base}/v1/metrics"), failure),
                                                         export_interval_millis=_metric_interval()))
        self.metrics = MeterProvider(resource=resource, metric_readers=readers)
        self.tracer = self.traces.get_tracer("wpt-telemetry")
        self.logger = self.logs.get_logger("wpt-telemetry")
        self.meter = self.metrics.get_meter("wpt-telemetry")
        self.gauges = {}
        self.open = [self.traces, self.logs, self.metrics]  # the providers not yet shut down

    def start(self, name, parent, attributes):
        if parent is None:
            context = Context()
        elif isinstance(parent, SpanContext):
            context = trace.set_span_in_context(NonRecordingSpan(parent))
        else:
            context = trace.set_span_in_context(parent)
        return self.tracer.start_span(name, context=context, attributes=_attributes(attributes))

    def end(self, span, attributes, error):
        span.set_attributes(_attributes(attributes))
        if error:
            span.set_status(Status(StatusCode.ERROR, error))
        span.end()

    def emit(self, span, name, text, attributes, error):
        self.logger.emit(timestamp=time.time_ns(), context=trace.set_span_in_context(span) if span else Context(),
                         severity_number=SeverityNumber.ERROR if error else SeverityNumber.INFO,
                         severity_text="ERROR" if error else "INFO", body=text if text is not None else name,
                         attributes=_attributes(attributes), event_name=name)

    def gauge(self, name, value):
        if name not in self.gauges:
            self.gauges[name] = self.meter.create_gauge(name)
        self.gauges[name].set(value)

    def close(self):
        """Send what is left and stop; called again, it goes on with what an
        interruption left open."""
        if self.failure is not None:
            self.failure.closing = True
        while self.open:
            self.open.pop(0).shutdown()


class _Failure:
    """Shared by the three exporters of one run: the first failure gets one
    warning; every later batch is still sent, and a failing one is dropped
    without a word. Once the run ends (closing), a kind of data whose send
    fails sends nothing more: the queued batches would each wait for the
    exporter's timeout, so the end of a run waits at most about one timeout
    per kind, however many batches a backend that does not answer left."""

    def __init__(self, endpoint, echo):
        self.endpoint, self.echo, self.warned = endpoint, echo, False
        self.closing, self.given_up = False, set()
        self.lock = threading.Lock()  # the three exporters send from their own threads

    def send(self, export, batch, success, failure):
        if self.closing and success in self.given_up:  # success: which kind of data this is
            return failure
        try:
            with _quiet():
                result = export(batch)
            reason = None if result == success else "it did not accept the data"
        except Exception as error:  # an exporter may raise on a network error
            result, reason = failure, f"{type(error).__name__}: {error}"
        if result != success:
            with self.lock:
                warned, self.warned = self.warned, True
                if self.closing:
                    self.given_up.add(success)
            if not warned:
                self.echo(f"warning: the OpenTelemetry backend at {self.endpoint} failed ({reason}); the run goes "
                          "on with its record in the store, and the live data is sent again quietly while it fails",
                          True)
        return result


# The OTLP exporters log a message on every batch that fails, and _Failure
# already gave one warning. Their loggers get a filter that drops a message
# only while one of WPT's exporters is sending from the same thread: any
# other use of these exporters in the process keeps its messages, and no
# logger's level changes.
_EXPORTER_LOGGERS = ("opentelemetry.exporter.otlp.proto.http.trace_exporter",
                     "opentelemetry.exporter.otlp.proto.http._log_exporter",
                     "opentelemetry.exporter.otlp.proto.http.metric_exporter")
_sending = threading.local()


class _WhileWptSends(logging.Filter):
    def filter(self, record):
        return not getattr(_sending, "active", False)


_QUIET_FILTER = _WhileWptSends()


def _quiet_exporters():
    for name in _EXPORTER_LOGGERS:
        logging.getLogger(name).addFilter(_QUIET_FILTER)  # once: a filter already there is not added again


@contextlib.contextmanager
def _quiet():
    _sending.active = True
    try:
        yield
    finally:
        _sending.active = False


class _Spans(SpanExporter):
    def __init__(self, inner, failure):
        self.inner, self.failure = inner, failure

    def export(self, spans):
        return self.failure.send(self.inner.export, spans, SpanExportResult.SUCCESS, SpanExportResult.FAILURE)

    def shutdown(self):
        with _quiet():
            self.inner.shutdown()

    def force_flush(self, timeout_millis=30000):
        return True


class _Logs(LogRecordExporter):
    def __init__(self, inner, failure):
        self.inner, self.failure = inner, failure

    def export(self, batch):
        return self.failure.send(self.inner.export, batch, LogRecordExportResult.SUCCESS, LogRecordExportResult.FAILURE)

    def shutdown(self):
        with _quiet():
            self.inner.shutdown()

    def force_flush(self, timeout_millis=30000):
        return True


class _Metrics(MetricExporter):
    def __init__(self, inner, failure):
        super().__init__(preferred_temporality=inner._preferred_temporality,
                         preferred_aggregation=inner._preferred_aggregation)
        self.inner, self.failure = inner, failure

    def export(self, metrics_data, timeout_millis=10_000, **kwargs):
        return self.failure.send(self.inner.export, metrics_data, MetricExportResult.SUCCESS,
                                 MetricExportResult.FAILURE)

    def shutdown(self, timeout_millis=30_000, **kwargs):
        with _quiet():
            self.inner.shutdown()

    def force_flush(self, timeout_millis=10_000):
        return True


def _interrupt_main():
    """Raise KeyboardInterrupt in the main thread, or run its own handler
    for Ctrl+C. On Linux and macOS, a real SIGINT to the main thread alone,
    which also ends a wait; only when Python handles SIGINT, since a SIGINT
    left to the system's default would kill the process without its
    RecordError. On Windows, which cannot send a signal to one thread,
    _thread.interrupt_main, which does nothing in that case either."""
    if not hasattr(signal, "pthread_kill"):
        _thread.interrupt_main()
    elif callable(signal.getsignal(signal.SIGINT)):
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)


def set_ref(store, kind, name, sha, telemetry):
    """store.set_ref(kind, name, sha), once the event "store.ref_set",
    naming the ref and the object it will point at, is in telemetry's
    record (RecordError otherwise, and the ref is not set)."""
    telemetry.irreversible("store.ref_set", ref=f"{kind}/{name}", object=sha)
    store.set_ref(kind, name, sha)


def short(value: str) -> str:
    """The first SHORT characters of a hash or a dataset id, for a record
    folder's name; ids.json keeps it whole."""
    return value[:SHORT]


def _endpoint(env_file, environ):
    environ = os.environ if environ is None else environ
    return environ.get(ENDPOINT) or read_env_file(env_file).get(ENDPOINT) or None


def _metric_interval():
    return float(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", METRIC_INTERVAL_MS))


def _attributes(attributes) -> dict:
    """OpenTelemetry attributes hold text, numbers and booleans: anything
    else goes as its JSON text."""
    return {key: value if isinstance(value, (str, bool, int, float)) else json.dumps(value, sort_keys=True, default=str)
            for key, value in attributes.items() if value is not None}


def _span_id(span) -> str:
    return f"{span.get_span_context().span_id:016x}"


def _print(text, error):
    print(text, file=sys.stderr if error else sys.stdout, flush=True)


def _json(document) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False, default=str) + "\n").encode()


def _iso(moment) -> str:
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _describe(error) -> str:
    if isinstance(error, SystemExit):
        return f"exit {error.code}" if not isinstance(error.code, str) else error.code
    return f"{type(error).__name__}: {error}"


def _traceback(error) -> str:
    """Where the error came from, as Python prints it, with the errors it
    was raised while handling (a refusal's sys.exit names its cause)."""
    return "".join(traceback.format_exception(error))


def _ending(error) -> tuple[str, int, str | None]:
    """(status, exit code, error) of a run that ended with error (None: it
    returned)."""
    if error is None:
        return "succeeded", 0, None
    if isinstance(error, SystemExit):
        if error.code in (0, None):
            return "succeeded", 0, None
        return "failed", error.code if isinstance(error.code, int) else 1, _describe(error)
    if isinstance(error, KeyboardInterrupt):
        return "failed", 130, "KeyboardInterrupt"
    return "failed", 1, _describe(error)


def _where() -> dict:
    """The tools, the code (wpt_utils.programs.code) and the repository
    commit; no machine name."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"tools": versions, "code": code(), **repository()}


def file_hashes(folder) -> dict:
    """{path under folder: sha256} of every file in it, with / between
    folder names: what a run read or wrote in a folder outside the store."""
    folder = Path(folder)
    return {path.relative_to(folder).as_posix(): sha256_file(path) for path in sorted(folder.rglob("*"))
            if path.is_file()}
