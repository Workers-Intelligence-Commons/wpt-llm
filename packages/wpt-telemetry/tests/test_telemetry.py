import contextlib
import http.server
import json
import logging
import signal
import socket
import threading
import time

import pytest

import wpt_telemetry
from wpt_store import LocalStore, open_store
from wpt_telemetry import ENDPOINT, FLUSH_THREAD, RecordError, Telemetry


def record(store, folder, name):
    return json.loads(store.read_record(folder, name))


def events(store, folder):
    return [json.loads(line) for line in store.read_record(folder, "events.jsonl").splitlines()]


def a_short_run(store, tmp_path, environ):
    with Telemetry.start(store, "wpt-example", ["d"], "d", inputs={"recipe": "d"}, env_file=tmp_path / ".env",
                         environ=environ) as telemetry:
        with telemetry.span("training", steps=2):
            for step in (1, 2):
                telemetry.event("training.step", values={"train_loss": 3.0 - step}, step=step)
        telemetry.log("saved model")
        telemetry.output(weights={"model.safetensors": "ab" * 32})
    return telemetry.folder


def test_a_local_record_holds_its_events_as_one_compressed_part_once_the_run_ended(tmp_path):
    store = open_store((tmp_path / "store").as_uri())
    with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
        telemetry.event("one")
        # while the run goes, its events are in the plain file, whole after a crash
        assert any(key.endswith("events.jsonl.open") for key in store.kind.list(f"records/{telemetry.folder}/"))
        assert [e["name"] for e in events(store, telemetry.folder) if e.get("kind") == "event"] == ["one"]
    stored = store.kind.list(f"records/{telemetry.folder}/")
    assert not any(key.endswith(".open") for key in stored)
    assert len([key for key in stored if f"/events.jsonl/" in key]) == 1
    assert record(store, telemetry.folder, "end.json")["status"] == "succeeded"
    assert [e["name"] for e in events(store, telemetry.folder) if e.get("kind") == "event"] == ["one"]


def test_a_backend_that_cannot_be_reached_gets_one_warning_and_the_run_goes_on(tmp_path, capsys, monkeypatch):
    # the live view is a convenience: an unreachable backend must neither stop
    # the run nor take anything from its record in the store
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")  # seconds the exporter waits before giving up
    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        closed = probe.getsockname()[1]
    store = open_store((tmp_path / "store").as_uri())
    folder = a_short_run(store, tmp_path, {ENDPOINT: f"http://127.0.0.1:{closed}"})

    assert capsys.readouterr().err.count("warning: the OpenTelemetry backend") == 1
    assert record(store, folder, "end.json")["status"] == "succeeded"
    steps = [e for e in events(store, folder) if e.get("name") == "training.step"]
    assert [(e["attributes"]["step"], e["values"]["train_loss"]) for e in steps] == [(1, 2.0), (2, 1.0)]


def test_the_exporters_own_messages_are_silenced_only_while_wpt_sends(tmp_path, capsys, caplog, monkeypatch):
    # WPT's one warning replaces the exporter's message on every batch, but
    # the same exporter used by other code in the process keeps its messages
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = probe.getsockname()[1]
    store = open_store((tmp_path / "store").as_uri())
    with caplog.at_level(logging.WARNING, logger="opentelemetry.exporter.otlp"):
        a_short_run(store, tmp_path, {ENDPOINT: f"http://127.0.0.1:{closed}"})
        assert [r for r in caplog.records if r.name.startswith("opentelemetry.exporter.otlp")] == []
        assert logging.getLogger("opentelemetry.exporter.otlp").level == logging.WARNING  # as caplog set it
        other = OTLPSpanExporter(f"http://127.0.0.1:{closed}/v1/traces")
        other.shutdown()
        other.export([])  # an exporter already shut down says so
    assert any(r.name.startswith("opentelemetry.exporter.otlp") for r in caplog.records)


class Collector(http.server.BaseHTTPRequestHandler):
    """Answers OTLP/HTTP requests as a backend does, keeping their paths."""
    received = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        Collector.received.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *_):
        pass


def test_the_backend_named_in_dot_env_receives_the_spans_logs_and_values(tmp_path, capsys):
    # the endpoint comes from .env like WPT_STORE, and the three kinds of data
    # reach the backend by the time the run ends
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (tmp_path / ".env").write_text(f"{ENDPOINT}=http://127.0.0.1:{server.server_address[1]}\n")
    store = open_store((tmp_path / "store").as_uri())
    try:
        a_short_run(store, tmp_path, {})
    finally:
        server.shutdown()

    assert {"/v1/traces", "/v1/logs", "/v1/metrics"} <= set(Collector.received)
    assert "warning" not in capsys.readouterr().err


class FailsFirst(Collector):
    """A backend whose first answer is an error, as one that is starting."""
    answered = 0

    def do_POST(self):
        FailsFirst.answered += 1
        if FailsFirst.answered > 1:
            return super().do_POST()
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(400)  # not one the exporter tries again on its own
        self.end_headers()


def test_after_a_backend_failure_the_live_data_is_still_sent(tmp_path, capsys):
    # a backend that failed once gets one warning, and what comes later still
    # reaches it, so the live view resumes when the backend is back
    Collector.received = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FailsFirst)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    store = open_store((tmp_path / "store").as_uri())
    try:
        a_short_run(store, tmp_path, {ENDPOINT: f"http://127.0.0.1:{server.server_address[1]}"})
    finally:
        server.shutdown()

    assert capsys.readouterr().err.count("warning: the OpenTelemetry backend") == 1
    assert len(Collector.received) == FailsFirst.answered - 1 >= 2  # every later batch was sent and accepted


def test_a_failure_is_recorded_with_its_traceback(tmp_path):
    # afterwards one must see where a failed run broke, not only its message
    store = open_store((tmp_path / "store").as_uri())

    def load_the_recipe():
        raise ValueError("no recipe named d")

    with pytest.raises(ValueError):
        with Telemetry.start(store, "wpt-example", ["d"], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
            with telemetry.span("inputs"):
                load_the_recipe()

    [span_end] = [e for e in events(store, telemetry.folder) if e["kind"] == "span_end" and e["name"] == "inputs"]
    end = record(store, telemetry.folder, "end.json")
    assert end["status"] == "failed" and end["error"] == "ValueError: no recipe named d"
    for text in (span_end["traceback"], end["traceback"]):
        assert "in load_the_recipe" in text and text.rstrip().endswith("ValueError: no recipe named d")



def test_each_thread_records_under_its_own_spans(tmp_path):
    # work run on other threads at once must land under its own item's span, never under whatever span another
    # thread has open; a span begun on a worker can be ended by the thread that puts the results together
    store = open_store((tmp_path / "store").as_uri())
    with Telemetry.start(store, "wpt-example", ["d"], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
        with telemetry.span("documents"):
            parent, begun, ready = telemetry.current(), {}, threading.Barrier(3)

            def work(item):
                with telemetry.within(parent):
                    begun[item] = span = telemetry.begin("source", source=item)
                with telemetry.within(span):
                    ready.wait()  # every worker has its span open: an event landing in another's would show
                    telemetry.event("converted", source=item)

            def unhanded():
                telemetry.event("outside")

            workers = [threading.Thread(target=work, args=(item,)) for item in ("a", "b", "c")]
            workers.append(threading.Thread(target=unhanded))
            for thread in workers:
                thread.start()
            for thread in workers:
                thread.join()
            for item in ("a", "b", "c"):
                with telemetry.finish(begun[item]) as span:
                    span.set(kept=1)
                    telemetry.event("kept", source=item)

    lines = events(store, telemetry.folder)
    [documents] = [e["span"] for e in lines if e["kind"] == "span_start" and e["name"] == "documents"]
    starts = {e["attributes"]["source"]: e for e in lines if e["kind"] == "span_start" and e["name"] == "source"}
    ends = {e["span"]: e for e in lines if e["kind"] == "span_end" and e["name"] == "source"}
    for item, start in starts.items():
        assert start["parent"] == documents
        assert [(e["name"], e["attributes"]["source"]) for e in lines
                if e["kind"] == "event" and e["span"] == start["span"]] == [("converted", item), ("kept", item)]
        assert ends[start["span"]]["attributes"] == {"kept": 1}
    [outside] = [e for e in lines if e["kind"] == "event" and e["name"] == "outside"]
    [run] = [e["span"] for e in lines if e["kind"] == "span_start" and e["parent"] is None]
    assert outside["span"] == run  # a thread given no span records under the run's own

def test_a_launcher_never_writes_back_an_older_copy_of_the_ids(tmp_path):
    # wpt-train's launcher opens the record, its training process joins it and
    # adds the ids it wrote; whatever the launcher adds later keeps them
    store = open_store((tmp_path / "store").as_uri())
    with Telemetry.start(store, "wpt-train", [], "d", inputs={"recipe": "d"}, env_file=tmp_path / ".env",
                         environ={}) as launcher:
        environment = launcher.child_environment()
        with Telemetry.join(open_store((tmp_path / "store").as_uri()), "wpt-train", environment,
                            env_file=tmp_path / ".env") as child:
            child.output(weights={"model.safetensors": "ab" * 32})
        launcher.input(restarts=0)

    assert record(store, launcher.folder, "ids.json") == {
        "inputs": {"recipe": "d", "restarts": 0}, "outputs": {"weights": {"model.safetensors": "ab" * 32}}}
    assert launcher.folder.split("-")[2:4] == ["train", "d"]  # <time>-wpt-train-<main id>-<random>


def test_on_s3_each_span_is_sent_when_it_ends_and_a_refused_send_stops_the_run(tmp_path, monkeypatch):
    # an S3 store sends lines in groups; a span's lines must be in the bucket
    # once it ends, and a group the bucket refuses must stop the run like any
    # other part of the record
    import boto3
    from moto import mock_aws

    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/records")
        with pytest.raises(RecordError):
            with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
                with telemetry.span("training"):
                    telemetry.event("training.step", values={"train_loss": 2.0}, step=1)
                ended = [e["kind"] for e in events(open_store("s3://wpt-test/records"), telemetry.folder)]
                assert ended[-3:] == ["span_start", "event", "span_end"]

                def refused(**request):
                    raise OSError("the bucket refused the request")

                monkeypatch.setattr(store.client, "put_object", refused)
                with telemetry.span("saving"):
                    pass
                telemetry.log("never reached")
        assert "end.json" not in store.record_files(telemetry.folder)


def refused_in_the_background(monkeypatch):
    """The store refuses every send made by the flush thread, which now
    tries every few hundredths of a second; the command's own sends pass."""
    real = LocalStore._raw_flush_records

    def refuse(self):
        if threading.current_thread().name == FLUSH_THREAD:
            raise OSError(28, "No space left on device")
        real(self)

    monkeypatch.setattr(wpt_telemetry, "FLUSH_SECONDS", 0.02)
    monkeypatch.setattr(LocalStore, "_raw_flush_records", refuse)


def test_a_record_that_broke_in_the_background_ends_the_run_in_record_error(tmp_path, monkeypatch):
    # a process that ignores Ctrl+C is not interrupted; even when its with
    # block then ends without an error, the run must end in RecordError (exit
    # code 3), never as a run that succeeded without its end.json
    refused_in_the_background(monkeypatch)
    store = open_store((tmp_path / "store").as_uri())
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with pytest.raises(RecordError, match="No space left on device"):
            with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
                time.sleep(0.5)  # the flush thread fails meanwhile
    finally:
        signal.signal(signal.SIGINT, previous)
    assert "end.json" not in store.record_files(telemetry.folder)


def test_a_send_refused_in_the_background_interrupts_the_run_before_its_next_record_call(tmp_path, monkeypatch):
    # the run must stop when the send fails, not at its next record call,
    # and as a record failure, not as a stop the user asked for
    refused_in_the_background(monkeypatch)
    store = open_store((tmp_path / "store").as_uri())
    started = time.monotonic()
    with pytest.raises(RecordError, match="No space left on device"):
        with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}):
            while time.monotonic() - started < 30:  # work that records nothing
                pass
    assert time.monotonic() - started < 10


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="Windows interrupts only between Python lines")
@pytest.mark.parametrize("waiting", ["sleep", "subprocess"])
def test_a_send_refused_in_the_background_ends_a_wait_of_the_main_thread(tmp_path, monkeypatch, waiting):
    # a command waiting (a launcher on its training processes, a pause) must
    # stop when the send fails too, not once its wait ends
    import subprocess
    import sys

    refused_in_the_background(monkeypatch)
    store = open_store((tmp_path / "store").as_uri())
    started = time.monotonic()
    with pytest.raises(RecordError, match="No space left on device"):
        with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}):
            if waiting == "sleep":
                time.sleep(30)
            else:
                with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) as child:
                    try:
                        child.wait()
                    finally:
                        child.kill()
    assert time.monotonic() - started < 10


def test_a_ctrl_c_while_a_broken_record_closes_still_ends_in_record_error(tmp_path, monkeypatch):
    # a record that broke must end the command with code 3 even when a stop
    # asked by the user reaches it while the record closes: never with 130
    store = open_store((tmp_path / "store").as_uri())
    real = LocalStore._raw_append_record

    def disk_full(self, folder, name, data):
        if b"refused" in data:
            raise OSError(28, "No space left on device")
        real(self, folder, name, data)

    stop_flushing = Telemetry._stop_flushing

    def ctrl_c(self):  # arrives once the flush thread has stopped
        stop_flushing(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full)
    monkeypatch.setattr(Telemetry, "_stop_flushing", ctrl_c)
    with pytest.raises(RecordError, match="No space left on device"):
        with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
            with contextlib.suppress(RecordError):  # the command went on; close() must still say the record broke
                telemetry.log("refused")
    assert "end.json" not in store.record_files(telemetry.folder)


def test_an_interruption_that_left_the_lock_taken_does_not_stop_the_record_from_closing(tmp_path):
    # an interruption can land just after the main thread took the lock and
    # leave it taken; close() must still return, not wait for that hold
    store = open_store((tmp_path / "store").as_uri())
    closed = threading.Event()

    def run():  # stands for the main thread; a daemon, so that a hang cannot hold the tests
        telemetry = Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={})
        telemetry._lock.acquire()  # never released
        telemetry.close()
        closed.set()

    threading.Thread(target=run, daemon=True).start()
    assert closed.wait(10)


def test_a_record_failure_while_the_record_starts_ends_in_record_error_not_a_stop(tmp_path, monkeypatch):
    # a background send that fails must never reach start() as a Ctrl+C:
    # that would end the command with 130, the code of a stop by the user
    refused_in_the_background(monkeypatch)
    real = LocalStore._raw_write_record

    def slow(self, folder, name, data):
        time.sleep(0.2)  # ten times the flush thread's period in this test
        real(self, folder, name, data)

    monkeypatch.setattr(LocalStore, "_raw_write_record", slow)
    store = open_store((tmp_path / "store").as_uri())
    with pytest.raises(RecordError, match="No space left on device"):
        with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}):
            time.sleep(1)


def test_a_bad_file_name_or_a_mixed_file_is_the_callers_error_and_the_record_goes_on(tmp_path):
    # a step or a test that names a file wrongly made a mistake; the record
    # itself is fine and the run must still end with its end.json
    from wpt_store import MixedRecord

    store = open_store((tmp_path / "store").as_uri())
    with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
        for name in ("results/a.json", ".hidden", "été.json"):
            with pytest.raises(ValueError, match="letters, digits"):
                telemetry.write_file(name, b"{}\n")
        telemetry.write_file("summary.json", b"{}\n")
        with pytest.raises(MixedRecord):
            telemetry.append_file("summary.json", b"{}\n")
        telemetry.log("still recorded")
    assert record(store, telemetry.folder, "end.json")["status"] == "succeeded"
    assert store.read_record(telemetry.folder, "summary.json") == b"{}\n"


def test_ctrl_c_still_ends_the_run_with_code_130(tmp_path):
    # a stop the user asked for is recorded as such, unlike a record failure
    store = open_store((tmp_path / "store").as_uri())
    with pytest.raises(KeyboardInterrupt):
        with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
            raise KeyboardInterrupt
    assert record(store, telemetry.folder, "end.json")["exit_code"] == 130


@pytest.mark.parametrize("where", ["local", "s3"])
def test_after_the_run_ends_every_recording_call_is_refused(tmp_path, monkeypatch, where):
    # a call made after the record is closed would be lost without a word:
    # it is a mistake in the command, and must fail loudly on either store
    with contextlib.ExitStack() as stack:
        if where == "s3":
            import boto3
            from moto import mock_aws

            for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test",
                                "AWS_DEFAULT_REGION": "us-east-1", "XDG_CACHE_HOME": str(tmp_path / "cache"),
                                "LOCALAPPDATA": str(tmp_path / "cache")}.items():
                monkeypatch.setenv(name, value)
            monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
            monkeypatch.delenv("AWS_PROFILE", raising=False)
            stack.enter_context(mock_aws())
            boto3.client("s3").create_bucket(Bucket="wpt-test")
            store = open_store("s3://wpt-test/records")
        else:
            store = open_store((tmp_path / "store").as_uri())
        with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env", environ={}) as telemetry:
            telemetry.log("during the run")
        written = store.read_record(telemetry.folder, "events.jsonl")

        calls = [lambda: telemetry.log("late"), lambda: telemetry.event("late"), lambda: telemetry.input(late=1),
                 lambda: telemetry.output(late=1), lambda: telemetry.span("late").__enter__(),
                 lambda: telemetry.write_file("late.json", b"{}\n"), lambda: telemetry.append_file("late.jsonl", b"\n"),
                 lambda: telemetry.irreversible("late"), telemetry.child_environment]
        for call in calls:
            with pytest.raises(RuntimeError, match="is closed"):
                call()
        assert store.read_record(telemetry.folder, "events.jsonl") == written


def test_the_end_of_a_run_waits_about_one_timeout_per_kind_for_a_backend_that_never_answers(tmp_path, monkeypatch):
    # docs/observability.md bounds the wait at the end of a run by the
    # exporter's timeout for each of the three kinds of data, however many
    # batches a silent backend left waiting
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(64)  # the system accepts the connections; nothing ever answers
    store = open_store((tmp_path / "store").as_uri())
    try:
        with Telemetry.start(store, "wpt-example", [], "d", env_file=tmp_path / ".env",
                             environ={ENDPOINT: f"http://127.0.0.1:{silent.getsockname()[1]}"}) as telemetry:
            for step in range(3000):
                telemetry.event("training.step", values={"train_loss": 1.0}, step=step)
            ending = time.monotonic()
        waited = time.monotonic() - ending
    finally:
        silent.close()
    assert waited < 3 * 1 + 2  # three kinds, one timeout each, and some slack
