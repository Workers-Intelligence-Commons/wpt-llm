"""Gpu workers the manager rents, a lost one replaced, and a manager that
comes back (docs/machines.md, "Rented gpu workers"). No provider is ever
called: a fake provider rents real gpu workers on this machine, and
SkyPilot is a fake `sky` command."""

import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import wpt_manager
from wpt_manager import RUN_FILE, Failed, main, rental, unfinished
from wpt_manager.machines import Workers
from wpt_manager.rental import Address, Provider, RentalError
from wpt_manager.skypilot import SkyPilot
from wpt_worker.link import Key


class Fake(Provider):
    """Rents gpu workers as processes of this machine, with the token the
    manager set; a rental of several is a cluster, its workers in the place
    of the rental's name at the machine's private address. With lose, the
    first worker it rents ends right after it gave its address, as a machine
    the provider takes back. failures: how many stops fail first."""

    cluster = True
    calls, processes, lose, failures = [], {}, False, 0

    def rent(self, name, token, interruptible, count=1):
        Fake.calls.append(("rent", name, interruptible) if count == 1 else ("rent", name, interruptible, count))
        found = []
        for index in range(count):
            folder = Path(self.settings["FAKE_FOLDER"]) / name / str(index)
            env = {**os.environ, "WPT_WORKER_TOKEN": token, "WPT_WORKER_PORT": "0",
                   "WPT_WORKER_ADDRESS": "127.0.0.1", "WPT_WORKER_FOLDER": str(folder)}
            if count > 1:
                env.update(WPT_WORKER_PLACE=name, WPT_WORKER_PRIVATE="127.0.0.1")
            process = subprocess.Popen([sys.executable, "-m", "wpt_worker"], stdout=subprocess.PIPE, text=True, env=env)
            found.append((process, process.stdout.readline(), token))
        Fake.processes[name] = found[0]
        Fake.cluster_of[name] = found

    def address(self, name):
        process, line, _ = Fake.processes[name]
        key = Key.parse(line)  # what the worker printed: its token hidden
        if Fake.lose and len(Fake.processes) == 1:
            process.kill()
            process.wait()
        return Address(key.host, key.port, key.pin)

    def addresses(self, name, count):
        keys = [Key.parse(line) for _, line, _ in Fake.cluster_of[name]]
        return [Address(key.host, key.port, key.pin, key.place, key.private) for key in keys]

    def stop(self, name):
        if Fake.failures:
            Fake.failures -= 1
            Fake.calls.append(("stop failed", name))
            raise RentalError("the provider does not answer")
        Fake.calls.append(("stop", name))
        for process, _, _ in Fake.cluster_of[name]:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


Fake.cluster_of = {}


@pytest.fixture
def fake(built, tmp_path, monkeypatch):
    Fake.calls, Fake.processes, Fake.lose, Fake.failures, Fake.cluster_of = [], {}, False, 0, {}
    monkeypatch.setattr(rental, "providers", lambda: {"fake": Fake})
    monkeypatch.setenv("WPT_RENTAL_PROVIDER", "fake")
    monkeypatch.setenv("FAKE_FOLDER", str(tmp_path / "rented"))
    monkeypatch.setenv("WPT_RECIPE", "d")
    yield Fake
    for group in Fake.cluster_of.values():
        for process, _, _ in group:
            if process.poll() is None:
                process.kill()
                process.wait()


def state(tmp_path, name) -> dict:
    return json.loads((tmp_path / "runs" / "manager-state" / f"{name}.json").read_text())


def test_the_manager_rents_its_gpu_workers_sets_their_tokens_never_shows_them_and_stops_them(built, fake, tmp_path,
                                                                                              monkeypatch, capfd):
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "rented")
    monkeypatch.setenv("WPT_RENTAL_KIND", "interruptible")
    main([])
    [(_, name, interruptible), stop] = fake.calls
    assert interruptible and stop == ("stop", name) and name.startswith("wpt-rented-")
    token = fake.processes[name][2]
    out = capfd.readouterr()
    assert token not in out.out + out.err and f"{token[:4]}********" in out.out
    for run in built.records("wpt-manager") + built.records("wpt-train"):
        assert token not in json.dumps(run.events) + json.dumps(run.start)
    [run] = built.records("wpt-manager")
    assert run.end["status"] == "succeeded"
    assert [e["attributes"]["rental"] for e in run.events if e.get("name") == "manager.rental_stopped"] == [name]
    # the workers' keys stay on the manager's disk, readable by its owner only; a run that ended has none left
    path = tmp_path / "runs" / "manager-state" / "rented.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and state(tmp_path, "rented")["workers"] == []
    assert built.store.get_ref("models", "rented") is not None


def test_a_lost_rented_worker_is_reported_replaced_and_the_run_resumes_from_the_store(built, fake, tmp_path,
                                                                                      monkeypatch):
    fake.lose = True
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "lost")
    main([])
    [first, second] = [call[1] for call in fake.calls if call[0] == "rent"]
    assert [call for call in fake.calls if call[0] == "stop"] == [("stop", first), ("stop", second)]
    [run] = built.records("wpt-manager")
    assert run.end["status"] == "succeeded"
    assert [e["attributes"]["rental"] for e in run.events if e.get("name") == "manager.worker_lost"] == [first]
    assert [(e["attributes"]["rental"], e["attributes"]["by"]) for e in run.events
            if e.get("name") == "manager.worker_replaced"] == [(first, second)]
    lost, resumed = built.records("wpt-train")
    assert lost.end["exit_code"] == 4 and resumed.end["status"] == "succeeded"
    assert "--resume" in resumed.start["arguments"] and "lost" in resumed.start["arguments"]


def test_a_run_stops_once_its_replacements_are_used_up_and_its_rentals_are_given_back(built, fake, tmp_path,
                                                                                      monkeypatch, capsys):
    fake.lose = True
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "spent")
    monkeypatch.setenv("WPT_REPLACEMENTS", "0")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 4
    [name] = [call[1] for call in fake.calls if call[0] == "rent"]
    assert ("stop", name) in fake.calls and "resumed 0 times already" in capsys.readouterr().err
    assert built.store.get_ref("models", "spent") is None


def test_a_lost_worker_a_person_started_is_not_replaced(built, fake, tmp_path, monkeypatch, capsys):
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    monkeypatch.setenv("WPT_GPU_WORKERS", str(Key("127.0.0.1", port, "0" * 64, "person-token")))
    monkeypatch.setenv("WPT_RUN", "given")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 4 and not fake.calls
    err = capsys.readouterr().err
    assert "lost gpu workers a person started" in err and "person-token" not in err


def test_bad_rental_settings_are_refused_before_anything_runs(built, fake, monkeypatch, capsys):
    for setting, value, said in (("WPT_RENTAL_KIND", "cheap", "one of on-demand, interruptible"),
                                 ("WPT_REPLACEMENTS", "many", "a number of times")):
        monkeypatch.setenv(setting, value)
        with pytest.raises(SystemExit) as refused:
            main([])
        assert refused.value.code == 2 and said in capsys.readouterr().err
        monkeypatch.delenv(setting)
    assert not built.records("wpt-manager") and not fake.calls


def test_a_manager_that_comes_back_goes_on_with_its_unfinished_run_on_its_workers_and_starts_no_other(
        built, fake, tmp_path, monkeypatch, capsys):
    # Docker starts the same container again after a crash: its run has a
    # record with no end, and its rented worker is still there; the manager
    # reconnects to it with the key its disk holds and goes on; with no
    # WPT_RUN it starts no other run
    monkeypatch.delenv("WPT_RUN", raising=False)
    provider = Fake(dict(os.environ), tmp_path)
    provider.rent("wpt-cut-1", "token-of-the-cut-run", False)
    found = provider.address("wpt-cut-1")
    key = Key(found.host, found.port, found.pin, "token-of-the-cut-run")
    kept = tmp_path / "runs" / "manager-state"
    workers = Workers.new(kept, "cut", "d")
    workers.state["workers"] = [{"key": str(key), "rental": {"provider": "fake", "name": "wpt-cut-1",
                                                             "kind": "on-demand"}}]
    workers.save()
    workers.release()  # the manager that ran it crashed: the system gave its lock back
    folder = "20261025T120000Z-wpt-manager-cut-0000aaaa"
    built.store.write_record(folder, "start.json", json.dumps({"command": "wpt-manager", "started": "2026-10-25T12:00:00Z",
                                                               "arguments": [], "inputs": {"run": "cut"}}).encode())
    built.store.write_record(folder, "ids.json", json.dumps({"inputs": {"run": "cut"}, "outputs": {}}).encode())
    built.store.append_record(folder, "events.jsonl", b"")
    assert unfinished(built.store, "cut")

    main([])
    assert not (tmp_path / RUN_FILE).exists()  # no new run was made
    cut, went_on = built.records("wpt-manager")
    assert cut.end is None and went_on.start["inputs"]["run"] == "cut" and went_on.end["status"] == "succeeded"
    assert [call for call in fake.calls if call[0] == "rent"] == [("rent", "wpt-cut-1", False)]  # no new rental
    assert ("stop", "wpt-cut-1") in fake.calls and not unfinished(built.store, "cut")
    assert "it goes on, reconnecting to its gpu workers" in capsys.readouterr().out


def test_skypilot_rents_with_the_token_in_its_environment_reads_the_address_and_stops(tmp_path, monkeypatch):
    # a fake sky command records each call and the token it sees
    bin_folder = tmp_path / "bin"
    bin_folder.mkdir()
    pin = "c" * 64
    calls = tmp_path / "calls.jsonl"
    (bin_folder / "sky").write_text(f"""#!{sys.executable}
import json, os, sys
arguments = sys.argv[1:]
with open({str(calls)!r}, "a") as out:
    out.write(json.dumps({{"arguments": arguments, "token": os.environ.get("WPT_WORKER_TOKEN"),
                          "cwd": os.getcwd(), "names": sorted(os.environ)}}) + "\\n")
if arguments[0] == "logs":
    print("installing...")
    print("wpt-worker://secr********@127.0.0.1:8470#{pin}")
elif arguments[0] == "status":
    print("http://34.0.0.9:30123")
elif arguments[0] == "down" and arguments[-1] == "wpt-gone":
    print("Cluster(s) not found: wpt-gone")
    sys.exit(1)
""")
    (bin_folder / "sky").chmod(0o755)
    settings = {**os.environ, "PATH": f"{bin_folder}{os.pathsep}{os.environ['PATH']}", "WPT_RENTAL_GPUS": "H100:8",
                "WPT_WORKER_WAIT": "600", "RUNPOD_API_KEY": "provider-key", "AWS_ACCESS_KEY_ID": "provider-key",
                "WPT_STORE": "s3://bucket", "WPT_STORE_ACCESS_KEY_ID": "store-key",
                "WPT_STORE_SECRET_ACCESS_KEY": "store-secret", "GITHUB_TOKEN": "x", "WPT_RECIPE": "d"}
    sky = SkyPilot(settings, tmp_path)
    sky.rent("wpt-one", "secret-token-123", True)
    assert sky.address("wpt-one") == Address("34.0.0.9", 30123, pin)
    sky.stop("wpt-one")
    sky.stop("wpt-gone")  # already given back: nothing to do
    seen = [json.loads(line) for line in calls.read_text().splitlines()]
    launch = seen[0]["arguments"]
    assert launch[:6] == ["launch", "-c", "wpt-one", "-y", "-d", "--down"]
    assert "--use-spot" in launch and launch[launch.index("--gpus") + 1] == "H100:8"
    assert ["--env", "WPT_WORKER_TOKEN"] == launch[launch.index("WPT_WORKER_TOKEN") - 1:launch.index("WPT_WORKER_TOKEN") + 1]
    assert "--env" in launch and "WPT_WORKER_WAIT" in launch and launch[-1] == "deploy/gpu-worker/skypilot.yaml"
    assert all("secret-token-123" not in " ".join(call["arguments"]) for call in seen)  # never on a command line
    assert seen[0]["token"] == "secret-token-123" and seen[0]["cwd"] == str(tmp_path)
    assert [call["arguments"][0] for call in seen] == ["launch", "logs", "status", "down", "down"]
    # sky gets its own provider keys, the shell's basics and what the workers are given, never the store's
    for call in seen:
        assert {"PATH", "HOME", "RUNPOD_API_KEY", "AWS_ACCESS_KEY_ID", "WPT_WORKER_WAIT"} <= set(call["names"])
        assert not ({"WPT_STORE", "WPT_STORE_ACCESS_KEY_ID", "WPT_STORE_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "WPT_RECIPE"}
                    & set(call["names"]))
    assert "--num-nodes" not in launch and not any(argument.startswith("WPT_WORKER_PLACE") for argument in launch)

    sky.rent("wpt-two", "t", False)
    assert "--use-spot" not in json.loads(calls.read_text().splitlines()[-1])["arguments"]
    with pytest.raises(RentalError, match="WPT_RENTAL_GPUS is not set"):
        SkyPilot({**settings, "WPT_RENTAL_GPUS": ""}, tmp_path).rent("wpt-three", "t", False)
    with pytest.raises(RentalError, match="sky is not installed"):
        SkyPilot({**settings, "PATH": str(tmp_path / "nowhere")}, tmp_path).stop("wpt-one")


def test_every_rental_is_tried_at_the_end_and_those_still_rented_are_named_and_kept(tmp_path, monkeypatch):
    # one stop that fails (sky down failing) never leaves the later rentals running
    from wpt_telemetry import Telemetry

    stopped = []

    class Stubborn(Provider):
        def stop(self, name):
            if name == "wpt-b":
                raise RentalError("sky down failed")
            stopped.append(name)

    monkeypatch.setattr(rental, "providers", lambda: {"stubborn": Stubborn})
    workers = Workers.new(tmp_path, "three", "d")
    entries = [{"key": None, "rental": {"provider": "stubborn", "name": name, "kind": "on-demand"}}
               for name in ("wpt-a", "wpt-b", "wpt-c")]
    workers.state["workers"] = entries
    with pytest.raises(RentalError, match=r"still rented: wpt-b \(sky down failed\)"):
        workers.stop({}, tmp_path, Telemetry.off())
    assert stopped == ["wpt-a", "wpt-c"]
    assert json.loads((tmp_path / "three.json").read_text())["workers"] == [entries[1]]  # kept, to stop again


def test_a_run_another_manager_runs_is_left_to_it(built, fake, tmp_path, monkeypatch, capsys):
    # two managers on one state folder: the second neither goes on with the
    # first one's run nor starts it, so it never gives back its rentals
    kept = tmp_path / "runs" / "manager-state"
    busy = Workers.new(kept, "busy", "d")  # the other manager, holding the run's lock
    busy.state["workers"] = [{"key": None, "rental": {"provider": "fake", "name": "wpt-busy-1", "kind": "on-demand"}}]
    busy.save()
    assert unfinished(built.store, "busy")  # no record yet: it looks cut short
    monkeypatch.setenv("WPT_RUN", "busy")
    with pytest.raises(SystemExit) as refused:
        main([])
    assert refused.value.code == 2 and "another wpt manager's" in capsys.readouterr().err
    assert not fake.calls and not built.records("wpt-manager")
    assert json.loads((kept / "busy.json").read_text())["workers"] == busy.state["workers"]
    busy.release()


def test_skypilot_rents_several_workers_as_one_cluster_and_reads_each_node_s_addresses(tmp_path):
    # one request with --num-nodes; the nodes' keys (with their private
    # addresses) and `wpt-node` lines come from the logs, the addresses to
    # reach them at from the SSH configuration sky writes under HOME
    bin_folder, home = tmp_path / "bin", tmp_path / "home"
    (home / ".sky" / "generated" / "ssh").mkdir(parents=True)
    bin_folder.mkdir()
    pins = ["a" * 64, "b" * 64]
    (bin_folder / "sky").write_text(f"""#!{sys.executable}
import json, sys
with open({str(tmp_path / "calls.jsonl")!r}, "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "logs":
    print("wpt-node 0 10.0.0.1")
    print("wpt-worker://clus********@0.0.0.0:8470?place=wpt-big&private=10.0.0.1#{pins[0]}")
    print("wpt-node 1 10.0.0.2")
    print("wpt-worker://clus********@0.0.0.0:8470?place=wpt-big&private=10.0.0.2#{pins[1]}")
""")
    (bin_folder / "sky").chmod(0o755)
    settings = {**os.environ, "PATH": f"{bin_folder}{os.pathsep}{os.environ['PATH']}", "WPT_RENTAL_GPUS": "A100:4",
                "HOME": str(home), "WPT_RENTAL_WAIT": "5"}
    sky = SkyPilot(settings, tmp_path)
    assert SkyPilot.cluster and not Provider.cluster
    sky.rent("wpt-big", "token", False, 2)
    launch = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert launch[launch.index("--num-nodes") + 1] == "2" and "WPT_WORKER_PLACE=wpt-big" in launch
    config = home / ".sky" / "generated" / "ssh" / "wpt-big"
    assert sky._cluster("wpt-big", 2) is None  # no node has an address to be reached at yet
    config.write_text("Host wpt-big\n  HostName 34.0.0.1\nHost wpt-big-worker1\n  HostName 34.0.0.2\n")
    assert sky.addresses("wpt-big", 2) == [Address("34.0.0.1", 8470, pins[0], "wpt-big", "10.0.0.1"),
                                          Address("34.0.0.2", 8470, pins[1], "wpt-big", "10.0.0.2")]


def test_several_workers_are_rented_as_one_cluster_in_one_request_and_a_provider_that_cannot_rents_one_each(
        built, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("WPT_GPU_WORKERS", "2")
    monkeypatch.setenv("WPT_STRATEGY", "ddp")  # several machines train as one synced run
    monkeypatch.setenv("WPT_RUN", "pair")
    main([])
    [(_, name, _, count), stop] = fake.calls
    assert count == 2 and stop == ("stop", name)  # one request, one stop
    [run] = built.records("wpt-manager")
    [rent] = [e for e in run.events if e["kind"] == "span_start" and e["name"] == "manager.rent"]
    assert rent["attributes"]["workers"] == 2 and rent["attributes"]["rental"] == name
    [train] = built.records("wpt-train")
    [job] = [e for e in train.events if e["kind"] == "span_start" and e["name"] == "gpu_worker.job"]
    assert job["attributes"]["workers"] == 2

    fake.calls.clear()
    monkeypatch.setattr(Fake, "cluster", False)
    monkeypatch.setenv("WPT_RUN", "alone")
    main([])
    assert [call[:3] for call in fake.calls if call[0] == "rent"] == [("rent", fake.calls[0][1], False),
                                                                      ("rent", fake.calls[1][1], False)]
    assert len({call[1] for call in fake.calls if call[0] == "rent"}) == 2 and all(len(c) == 3 for c in fake.calls if c[0] == "rent")


class Recorder:
    """The telemetry methods Workers calls, recording what they are given."""

    def __init__(self):
        self.events, self.logs = [], []

    def event(self, name, /, values=None, **attributes):
        self.events.append((name, attributes))

    irreversible = event

    def log(self, text, /, error=False, **attributes):
        self.logs.append(text)

    def span(self, name, /, **attributes):
        from contextlib import nullcontext
        from types import SimpleNamespace

        return nullcontext(SimpleNamespace(set=lambda **_: None))


def test_a_lost_worker_of_a_cluster_replaces_the_whole_cluster_and_a_crash_with_none_lost_replaces_none(
        fake, tmp_path, monkeypatch):
    monkeypatch.setenv("WPT_RENTAL_GPUS", "x")
    settings = dict(os.environ)
    workers = Workers.new(tmp_path / "state", "pair", "d")
    recorder = Recorder()
    workers.prepare({**settings, "WPT_GPU_WORKERS": "2"}, tmp_path, recorder)
    [first] = [call[1] for call in fake.calls if call[0] == "rent"]
    keys = workers.keys()
    assert [(key.place, key.private) for key in keys] == [(first, "127.0.0.1"), (first, "127.0.0.1")]
    assert keys[0].token == keys[1].token  # one rental, one token

    # a training process crashed: every worker still answers, so none is replaced and the run resumes
    assert workers.recover(settings, tmp_path, recorder)
    assert [call[0] for call in fake.calls] == ["rent"] and workers.state["resumed"] == 1
    assert "every gpu worker answers again" in recorder.logs[-1]

    # one worker of the cluster is lost: the whole cluster is given back and rented again
    process = fake.cluster_of[first][1][0]
    process.kill()
    process.wait()
    assert workers.recover(settings, tmp_path, recorder)
    [_, stop, second] = fake.calls
    assert stop == ("stop", first) and second[0] == "rent" and second[3] == 2
    assert [name for name, _ in recorder.events if name == "manager.worker_lost"] == ["manager.worker_lost"]
    [(_, replaced)] = [event for event in recorder.events if event[0] == "manager.worker_replaced"]
    assert replaced["rental"] == first and replaced["by"] == second[1] and replaced["workers"] == 2
    assert {key.place for key in workers.keys()} == {second[1]} and workers.state["resumed"] == 2
    workers.stop(settings, tmp_path, recorder)
    workers.release()


def test_a_crash_of_a_training_process_on_several_workers_resumes_the_run_within_the_same_limit_of_tries(
        built, fake, tmp_path, monkeypatch, capsys):
    real = wpt_manager.run_command
    crashes = [5]
    tried = []

    def crashing(step, arguments, log, last_line=False, environ=None):
        if arguments[0] == "wpt-train":
            tried.append(list(arguments))
            if crashes:
                raise Failed(step, crashes.pop())
        return real(step, arguments, log, last_line, environ)

    monkeypatch.setattr(wpt_manager, "run_command", crashing)
    monkeypatch.setenv("WPT_GPU_WORKERS", "2")
    monkeypatch.setenv("WPT_STRATEGY", "ddp")
    monkeypatch.setenv("WPT_RUN", "crashy")
    main([])
    assert len(tried) == 2 and tried[0] == tried[1] and "--resume" in tried[0]
    assert built.store.get_ref("models", "crashy") is not None
    assert [call[0] for call in fake.calls] == ["rent", "stop"]  # nothing was lost: no new rental
    assert "the run resumes from the last checkpoint in the store" in capsys.readouterr().out

    crashes[:] = [5, 5]  # the same limit as for a lost worker: WPT_REPLACEMENTS
    tried.clear()
    monkeypatch.setenv("WPT_RUN", "crashier")
    monkeypatch.setenv("WPT_REPLACEMENTS", "1")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 5 and len(tried) == 2
    assert "resumed 1 times already" in capsys.readouterr().err and built.store.get_ref("models", "crashier") is None


def test_rentals_that_could_not_be_stopped_are_tried_again_at_each_start_and_every_few_minutes_each_try_recorded(
        built, fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(wpt_manager, "SWEEP", 0.01)
    fake.failures = 2  # the stop at the run's end, then the first try of the sweep
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "sweep")
    main([])
    [name] = [call[1] for call in fake.calls if call[0] == "rent"]
    assert [call[0] for call in fake.calls] == ["rent", "stop failed", "stop failed", "stop"]
    out = capsys.readouterr().out
    assert "1 rental(s) are still rented" in out
    run, first, second = built.records("wpt-manager")
    assert run.end["status"] == "succeeded"
    assert [e["attributes"]["rental"] for e in run.events if e.get("name") == "manager.rental_stop_failed"] == [name]
    # each try of the sweep is a record of its own, and the one that worked says so
    assert first.start["inputs"] == second.start["inputs"] == {"stop_rentals": ["sweep"]}
    assert [e["attributes"]["rental"] for e in first.events if e.get("name") == "manager.rental_stop_failed"] == [name]
    assert [e["attributes"]["rental"] for e in second.events if e.get("name") == "manager.rental_stopped"] == [name]
    assert state(tmp_path, "sweep")["workers"] == []

    # a rental an earlier manager could not stop is tried again when the next one starts
    left = Workers.new(tmp_path / "runs" / "manager-state", "old", "d")
    left.state["workers"] = [{"key": None, "rental": {"provider": "fake", "name": "wpt-old-1", "kind": "on-demand"}}]
    left.end()
    left.release()
    fake.processes["wpt-old-1"] = None
    fake.cluster_of["wpt-old-1"] = []
    fake.calls.clear()
    monkeypatch.setenv("WPT_RUN", "next")
    main([])
    assert fake.calls[0] == ("stop", "wpt-old-1") and state(tmp_path, "old")["workers"] == []


def test_the_evaluation_runs_on_the_rented_workers_before_they_are_given_back(built, fake, tmp_path, monkeypatch):
    real = wpt_manager.run_command
    order = []

    def watching(step, arguments, log, last_line=False, environ=None):
        if arguments[0] == "wpt-eval":
            order.append(("eval", list(fake.calls), environ))
            return None
        return real(step, arguments, log, last_line, environ)

    monkeypatch.setattr(wpt_manager, "run_command", watching)
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "judged")
    monkeypatch.setenv("WPT_EVALUATE", "1")
    main([])
    [(_, calls, environ)] = order
    [(_, name, _)] = calls  # rented, not yet stopped
    assert Key.parse(environ["WPT_GPU_WORKERS"]).host == "127.0.0.1" and fake.calls[-1] == ("stop", name)


def test_a_failed_run_is_continued_by_the_same_command_under_the_name_wpt_run_txt_holds(built, tmp_path,
                                                                                         monkeypatch, capsys):
    real = wpt_manager.run_command
    fail = [True]

    def failing(step, arguments, log, last_line=False, environ=None):
        if arguments[0] == "wpt-train" and fail:
            fail.pop()
            raise Failed(step, 1)
        return real(step, arguments, log, last_line, environ)

    monkeypatch.setattr(wpt_manager, "run_command", failing)
    monkeypatch.setenv("WPT_RECIPE", "d")
    monkeypatch.delenv("WPT_RUN", raising=False)
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 1
    written = (tmp_path / RUN_FILE).read_text()
    [name] = [line.split()[2] for line in written.splitlines() if line.startswith("# run ")]
    assert built.store.get_ref("models", name) is None

    main([])  # the same command again: the same run, from what was kept
    assert (tmp_path / RUN_FILE).read_text() == written
    assert built.store.get_ref("models", name) is not None
    assert f"run {name} (named in" in capsys.readouterr().out

    main([])  # its model is in the store now: a new run under a new name
    assert (tmp_path / RUN_FILE).read_text() != written
    # another recipe's run name is never taken over
    (tmp_path / RUN_FILE).write_text("# run other-20260101T000000Z of the recipe not-d; once it is done\n")
    assert wpt_manager.previous_run(tmp_path, "d", built.store) is None
    assert wpt_manager.previous_run(tmp_path, "not-d", built.store) == "other-20260101T000000Z"


def test_in_the_container_a_run_that_failed_or_was_refused_exits_with_0_and_says_why(built, tmp_path, monkeypatch,
                                                                                    capsys):
    # restart: on-failure then restarts only a crash or a Docker restart
    monkeypatch.setenv("WPT_MANAGER_CONTAINER", "1")
    monkeypatch.delenv("WPT_RECIPE", raising=False)
    main([])  # returns: no SystemExit
    err = capsys.readouterr().err
    assert "WPT_RECIPE is not set" in err and "exits with 0" in err

    monkeypatch.setenv("WPT_RECIPE", "d")

    def failing(step, arguments, log, last_line=False, environ=None):
        raise Failed(step, 3)

    monkeypatch.setattr(wpt_manager, "run_command", failing)
    main([])
    assert "wpt-manager stopped with exit code 3" in capsys.readouterr().err
    [run] = built.records("wpt-manager")
    assert run.end["exit_code"] == 3  # the record keeps the real code

    monkeypatch.delenv("WPT_MANAGER_CONTAINER")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 3
