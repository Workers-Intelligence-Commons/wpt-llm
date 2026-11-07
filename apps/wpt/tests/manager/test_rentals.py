"""Gpu workers the manager rents, a lost one replaced, and a manager that
comes back (docs/machines.md, "Rented gpu workers"). No provider is ever
called: a fake provider rents real gpu workers on this machine, and
SkyPilot is a fake `sky` command."""

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

import wpt.manager
from wpt.manager import RUN_FILE, Failed, main, rental, unfinished
from wpt.train import LOST, StoreCheckpoint
from wpt.manager.machines import Workers
from wpt.manager.rental import Address, Cap, Provider, RentalError
from wpt.manager.skypilot import SkyPilot
from wpt_training import load_state
from wpt_utils.lines import INFO, bare
from wpt_worker.link import Key
from wpt_store.models import has_model


class Fake(Provider):
    """Rents gpu workers as processes of this machine, with the token the
    manager set; a rental of several is a cluster, its workers in the place
    of the rental's name at the machine's private address. The fixture keeps
    them off the GPU: each stands for a machine of its own, and NCCL refuses
    two processes on one GPU. With lose, the
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

    def address(self, name, watch):
        process, line, _ = Fake.processes[name]
        key = Key.parse(bare(line))  # what the worker printed: its token hidden
        if Fake.lose and len(Fake.processes) == 1:
            process.kill()
            process.wait()
        return Address(key.host, key.port, key.pin)

    def addresses(self, name, count, watch):
        keys = [Key.parse(bare(line)) for _, line, _ in Fake.cluster_of[name]]
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
def fake(built, on_the_cpu, tmp_path, monkeypatch):
    Fake.calls, Fake.processes, Fake.lose, Fake.failures, Fake.cluster_of = [], {}, False, 0, {}
    monkeypatch.setattr(rental, "providers", lambda: {"fake": Fake})
    monkeypatch.setenv("WPT_RENTAL_PROVIDER", "fake")
    monkeypatch.setenv("FAKE_FOLDER", str(tmp_path / "rented"))
    monkeypatch.setenv("WPT_RECIPE", "d")
    monkeypatch.setenv("WPT_KERNELS", "pytorch")  # the fake's workers have no GPU
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
    for run in built.records("wpt manager") + built.records("wpt train"):
        assert token not in json.dumps(run.events) + json.dumps(run.start)
    [run] = built.records("wpt manager")
    assert run.end["status"] == "succeeded"
    assert [e["attributes"]["rental"] for e in run.events if e.get("name") == "manager.rental_stopped"] == [name]
    # the workers' keys stay on the manager's disk, readable by its owner only; a run that ended has none left
    path = tmp_path / "runs" / "manager-state" / "rented.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and state(tmp_path, "rented")["workers"] == []
    assert has_model(built.store, "rented")


def test_a_lost_rented_worker_is_reported_replaced_and_the_run_resumes_from_the_store(built, fake, tmp_path,
                                                                                      monkeypatch):
    fake.lose = True
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "lost")
    main([])
    [first, second] = [call[1] for call in fake.calls if call[0] == "rent"]
    assert [call for call in fake.calls if call[0] == "stop"] == [("stop", first), ("stop", second)]
    [run] = built.records("wpt manager")
    assert run.end["status"] == "succeeded"
    assert [e["attributes"]["rental"] for e in run.events if e.get("name") == "manager.worker_lost"] == [first]
    assert [(e["attributes"]["rental"], e["attributes"]["by"]) for e in run.events
            if e.get("name") == "manager.worker_replaced"] == [(first, second)]
    lost, resumed = built.records("wpt train")
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
    assert not has_model(built.store, "spent")


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
                                 ("WPT_REPLACEMENTS", "many", "a number of times"),
                                 ("WPT_STOP_DEADLINE", "soon", "a number of seconds above 0")):
        monkeypatch.setenv(setting, value)
        with pytest.raises(SystemExit) as refused:
            main([])
        assert refused.value.code == 2 and said in capsys.readouterr().err
        monkeypatch.delenv(setting)
    assert not built.records("wpt manager") and not fake.calls


def cut_run(built, tmp_path) -> Workers:
    """The run named cut, which a crash left with its rental wpt-cut-1 and
    a record with no end."""
    provider = Fake(dict(os.environ), tmp_path)
    provider.rent("wpt-cut-1", "token-of-the-cut-run", False)
    found = provider.address("wpt-cut-1", lambda: None)
    key = Key(found.host, found.port, found.pin, "token-of-the-cut-run")
    kept = tmp_path / "runs" / "manager-state"
    workers = Workers.new(kept, "cut", "d", "test-session")
    workers.state["workers"] = [{"key": str(key), "rental": {"provider": "fake", "name": "wpt-cut-1",
                                                             "kind": "on-demand"}}]
    workers.save()
    workers.release()  # the manager that ran it crashed: the system gave its lock back
    folder = "test-session/20261025T120000Z-wpt-manager-cut-0000aaaa"
    built.store.write_record(folder, "start.json", json.dumps({"command": "wpt manager", "started": "2026-10-25T12:00:00Z",
                                                               "arguments": [], "inputs": {"run": "cut"}}).encode())
    built.store.write_record(folder, "ids.json", json.dumps({"inputs": {"run": "cut"}, "outputs": {}}).encode())
    built.store.append_record(folder, "events.jsonl", b"")
    return workers


def test_a_manager_that_comes_back_goes_on_with_its_unfinished_run_on_its_workers_and_starts_no_other(
        built, fake, tmp_path, monkeypatch, capsys):
    # Docker starts the same container again after a crash: its run has a
    # record with no end, and its rented worker is still there; the manager
    # reconnects to it with the key its disk holds and goes on; with no
    # WPT_RUN it starts no other run
    monkeypatch.delenv("WPT_RUN", raising=False)
    cut_run(built, tmp_path)
    assert unfinished(built.store, "cut")

    main([])
    assert not (tmp_path / RUN_FILE).exists()  # no new run was made
    cut, went_on = built.records("wpt manager")
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
    print("2026-11-05 15:27:29 INFO wpt-worker://secr********@127.0.0.1:8470#{pin}")
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
    assert sky.address("wpt-one", lambda: None) == Address("34.0.0.9", 30123, pin)
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
    workers = Workers.new(tmp_path, "three", "d", "test-session")
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
    busy = Workers.new(kept, "busy", "d", "test-session")  # the other manager, holding the run's lock
    busy.state["workers"] = [{"key": None, "rental": {"provider": "fake", "name": "wpt-busy-1", "kind": "on-demand"}}]
    busy.save()
    assert unfinished(built.store, "busy")  # no record yet: it looks cut short
    monkeypatch.setenv("WPT_RUN", "busy")
    with pytest.raises(SystemExit) as refused:
        main([])
    assert refused.value.code == 2 and "another wpt manager's" in capsys.readouterr().err
    assert not fake.calls and not built.records("wpt manager")
    assert json.loads((kept / "busy.json").read_text())["workers"] == busy.state["workers"]
    busy.release()


def test_skypilot_writes_the_runpod_key_into_its_own_file_under_home_before_it_runs_sky(tmp_path):
    # SkyPilot enables RunPod only when ~/.runpod/config.toml holds the key; the file is the
    # manager's setting, replaced at each run, readable by its owner only, absent without the setting
    bin_folder, home = tmp_path / "bin", tmp_path / "home"
    bin_folder.mkdir()
    (home / ".runpod").mkdir(parents=True)
    (home / ".runpod" / "config.toml").write_text('[default]\napi_key = "old"\n')
    (bin_folder / "sky").write_text(f"#!{sys.executable}\n")
    (bin_folder / "sky").chmod(0o755)
    settings = {"PATH": f"{bin_folder}{os.pathsep}{os.environ['PATH']}", "HOME": str(home)}
    config = home / ".runpod" / "config.toml"

    SkyPilot(settings, tmp_path).stop("wpt-one")
    assert tomllib.loads(config.read_text()) == {"default": {"api_key": "old"}}  # no setting: left alone

    key = 'a"b\\c-key'
    SkyPilot({**settings, "RUNPOD_API_KEY": key}, tmp_path).stop("wpt-one")
    assert tomllib.loads(config.read_text(encoding="utf-8")) == {"default": {"api_key": key}}
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.parent.stat().st_mode) == 0o700
    assert [path.name for path in config.parent.iterdir()] == ["config.toml"]

    fresh = tmp_path / "fresh"
    SkyPilot({**settings, "HOME": str(fresh)}, tmp_path).stop("wpt-one")
    assert not (fresh / ".runpod").exists()  # no setting: nothing written


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
    print("2026-11-05 15:27:29 INFO wpt-worker://clus********@0.0.0.0:8470?place=wpt-big&private=10.0.0.1#{pins[0]}")
    print("wpt-node 1 10.0.0.2")
    print("2026-11-05 15:27:29 INFO wpt-worker://clus********@0.0.0.0:8470?place=wpt-big&private=10.0.0.2#{pins[1]}")
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
    assert sky.addresses("wpt-big", 2, lambda: None) == [Address("34.0.0.1", 8470, pins[0], "wpt-big", "10.0.0.1"),
                                          Address("34.0.0.2", 8470, pins[1], "wpt-big", "10.0.0.2")]


def test_several_workers_are_rented_as_one_cluster_in_one_request_and_a_provider_that_cannot_rents_one_each(
        built, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("WPT_GPU_WORKERS", "2")
    monkeypatch.setenv("WPT_STRATEGY", "ddp")  # several machines train as one synced run
    monkeypatch.setenv("WPT_RUN", "pair")
    main([])
    [(_, name, _, count), stop] = fake.calls
    assert count == 2 and stop == ("stop", name)  # one request, one stop
    [run] = built.records("wpt manager")
    [rent] = [e for e in run.events if e["kind"] == "span_start" and e["name"] == "manager.rent"]
    assert rent["attributes"]["workers"] == 2 and rent["attributes"]["rental"] == name
    [train] = built.records("wpt train")
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

    def log(self, text, level=INFO, /, **attributes):
        self.logs.append(text)

    def span(self, name, /, **attributes):
        from contextlib import nullcontext
        from types import SimpleNamespace

        return nullcontext(SimpleNamespace(set=lambda **_: None))


def test_a_lost_worker_of_a_cluster_replaces_the_whole_cluster_and_a_crash_with_none_lost_replaces_none(
        fake, tmp_path, monkeypatch):
    monkeypatch.setenv("WPT_RENTAL_GPUS", "x")
    settings = dict(os.environ)
    workers = Workers.new(tmp_path / "state", "pair", "d", "test-session")
    recorder = Recorder()
    workers.prepare({**settings, "WPT_GPU_WORKERS": "2"}, tmp_path, recorder, lambda: None)
    [first] = [call[1] for call in fake.calls if call[0] == "rent"]
    keys = workers.keys()
    assert [(key.place, key.private) for key in keys] == [(first, "127.0.0.1"), (first, "127.0.0.1")]
    assert keys[0].token == keys[1].token  # one rental, one token

    # a training process crashed: every worker still answers, so none is replaced and the run resumes
    assert workers.recover(settings, tmp_path, recorder, lambda: None)
    assert [call[0] for call in fake.calls] == ["rent"] and workers.state["resumed"] == 1
    assert "every gpu worker answers again" in recorder.logs[-1]

    # one worker of the cluster is lost: the whole cluster is given back and rented again
    process = fake.cluster_of[first][1][0]
    process.kill()
    process.wait()
    assert workers.recover(settings, tmp_path, recorder, lambda: None)
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
    real = wpt.manager.run_command
    crashes = [5]
    tried = []

    def crashing(step, arguments, log, last_line=False, environ=None, stops=None):
        if arguments[0] == "train":
            tried.append(list(arguments))
            if crashes:
                raise Failed(step, crashes.pop())
        return real(step, arguments, log, last_line, environ)

    monkeypatch.setattr(wpt.manager, "run_command", crashing)
    monkeypatch.setenv("WPT_GPU_WORKERS", "2")
    monkeypatch.setenv("WPT_STRATEGY", "ddp")
    monkeypatch.setenv("WPT_RUN", "crashy")
    main([])
    assert len(tried) == 2 and tried[0] == tried[1] and "--resume" in tried[0]
    assert has_model(built.store, "crashy")
    assert [call[0] for call in fake.calls] == ["rent", "stop"]  # nothing was lost: no new rental
    assert "the run resumes from the last checkpoint in the store" in capsys.readouterr().out

    crashes[:] = [5, 5]  # the same limit as for a lost worker: WPT_REPLACEMENTS
    tried.clear()
    monkeypatch.setenv("WPT_RUN", "crashier")
    monkeypatch.setenv("WPT_REPLACEMENTS", "1")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 5 and len(tried) == 2
    assert "resumed 1 times already" in capsys.readouterr().err and not has_model(built.store, "crashier")


def test_rentals_that_could_not_be_stopped_are_tried_again_at_each_start_and_every_few_minutes_each_try_recorded(
        built, fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(wpt.manager, "SWEEP", 0.01)
    fake.failures = 2  # the stop at the run's end, then the first try of the sweep
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "sweep")
    main([])
    [name] = [call[1] for call in fake.calls if call[0] == "rent"]
    assert [call[0] for call in fake.calls] == ["rent", "stop failed", "stop failed", "stop"]
    out = capsys.readouterr().out
    assert "1 rental(s) are still rented" in out
    run, first, second = built.records("wpt manager")
    assert run.end["status"] == "succeeded"
    assert [e["attributes"]["rental"] for e in run.events if e.get("name") == "manager.rental_stop_failed"] == [name]
    # each try of the sweep is a record of its own, and the one that worked says so
    assert first.start["inputs"] == second.start["inputs"] == {"stop_rentals": ["sweep"]}
    assert [e["attributes"]["rental"] for e in first.events if e.get("name") == "manager.rental_stop_failed"] == [name]
    assert [e["attributes"]["rental"] for e in second.events if e.get("name") == "manager.rental_stopped"] == [name]
    assert state(tmp_path, "sweep")["workers"] == []

    # a rental an earlier manager could not stop is tried again when the next one starts
    left = Workers.new(tmp_path / "runs" / "manager-state", "old", "d", "test-session")
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
    real = wpt.manager.run_command
    order = []

    def watching(step, arguments, log, last_line=False, environ=None, stops=None):
        if arguments[0] == "eval":
            order.append(("eval", list(fake.calls), environ))
            return None
        return real(step, arguments, log, last_line, environ)

    monkeypatch.setattr(wpt.manager, "run_command", watching)
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "judged")
    monkeypatch.setenv("WPT_EVALUATE", "1")
    main([])
    [(_, calls, environ)] = order
    [(_, name, _)] = calls  # rented, not yet stopped
    assert Key.parse(environ["WPT_GPU_WORKERS"]).host == "127.0.0.1" and fake.calls[-1] == ("stop", name)


def test_training_on_rented_workers_names_the_triton_kernels_unless_the_settings_name_the_kernels(
        built, fake, tmp_path, monkeypatch):
    # docs/machines.md: the manager runs wpt train on the NVIDIA GPUs it
    # rents with WPT_KERNELS=triton; a WPT_KERNELS of the settings is left
    # as it is, and workers a person started get none from the manager
    real = wpt.manager.run_command
    named = []

    def watching(step, arguments, log, last_line=False, environ=None, stops=None):
        if arguments[0] == "train":
            named.append({**os.environ, **environ}.get("WPT_KERNELS"))  # what wpt train gets
            raise Failed(step, 2)
        return real(step, arguments, log, last_line, environ)

    monkeypatch.setattr(wpt.manager, "run_command", watching)
    monkeypatch.delenv("WPT_KERNELS")
    for run, workers, kernels in (("rented", "1", None), ("asked", "1", "tilelang"),
                                  ("started", str(Key("127.0.0.1", 9, "0" * 64, "person-token")), None)):
        monkeypatch.setenv("WPT_RUN", run)
        monkeypatch.setenv("WPT_GPU_WORKERS", workers)
        if kernels:
            monkeypatch.setenv("WPT_KERNELS", kernels)
        else:
            monkeypatch.delenv("WPT_KERNELS", raising=False)
        with pytest.raises(SystemExit):  # the training stood in for fails: the run stops there
            main([])
    assert named == ["triton", "tilelang", None]


def test_the_evaluation_on_rented_workers_names_the_triton_kernels_unless_the_settings_name_the_kernels(
        built, fake, tmp_path, monkeypatch):
    # docs/machines.md: wpt eval gets what wpt train gets, WPT_KERNELS=triton
    # on rented workers, a WPT_KERNELS of the settings as it is
    real = wpt.manager.run_command
    named = []

    def watching(step, arguments, log, last_line=False, environ=None, stops=None):
        if arguments[0] == "eval":
            named.append({**os.environ, **environ}.get("WPT_KERNELS"))  # what wpt eval gets
            return None
        if arguments[0] == "train":  # this machine may have no GPU for the Triton kernels: train with PyTorch's
            environ = {**environ, "WPT_KERNELS": "pytorch"}
        return real(step, arguments, log, last_line, environ)

    monkeypatch.setattr(wpt.manager, "run_command", watching)
    monkeypatch.setenv("WPT_EVALUATE", "1")
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    for run, kernels in (("rented", None), ("asked", "pytorch")):
        monkeypatch.setenv("WPT_RUN", run)
        if kernels:
            monkeypatch.setenv("WPT_KERNELS", kernels)
        else:
            monkeypatch.delenv("WPT_KERNELS", raising=False)
        main([])
    assert named == ["triton", "pytorch"]


def test_a_failed_run_is_continued_by_the_same_command_under_the_name_wpt_run_txt_holds(built, tmp_path,
                                                                                         monkeypatch, capsys):
    real = wpt.manager.run_command
    fail = [True]

    def failing(step, arguments, log, last_line=False, environ=None, stops=None):
        if arguments[0] == "train" and fail:
            fail.pop()
            raise Failed(step, 1)
        return real(step, arguments, log, last_line, environ)

    monkeypatch.setattr(wpt.manager, "run_command", failing)
    monkeypatch.setenv("WPT_RECIPE", "d")
    monkeypatch.delenv("WPT_RUN", raising=False)
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 1
    written = (tmp_path / RUN_FILE).read_text()
    [name] = [line.split()[2] for line in written.splitlines() if line.startswith("# run ")]
    assert not has_model(built.store, name)

    main([])  # the same command again: the same run, from what was kept
    assert (tmp_path / RUN_FILE).read_text() == written
    assert has_model(built.store, name)
    assert f"run {name} (named in" in capsys.readouterr().out

    main([])  # its model is in the store now: a new run under a new name
    assert (tmp_path / RUN_FILE).read_text() != written
    # another recipe's run name is never taken over
    (tmp_path / RUN_FILE).write_text("# run other-20260101T000000Z of the recipe not-d in session s1; once it is done\n")
    assert wpt.manager.previous_run(tmp_path, "d", built.store, None, None, None) is None
    assert wpt.manager.previous_run(tmp_path, "not-d", built.store, None, None, None) == ("other-20260101T000000Z", "s1")


def test_in_the_container_a_run_that_failed_or_was_refused_exits_with_0_and_says_why(built, tmp_path, monkeypatch,
                                                                                    capsys):
    # restart: on-failure then restarts only a crash or a Docker restart
    monkeypatch.setenv("WPT_MANAGER_CONTAINER", "1")
    monkeypatch.delenv("WPT_RECIPE", raising=False)
    main([])  # returns: no SystemExit
    err = capsys.readouterr().err
    assert "WPT_RECIPE is not set" in err and "exits with 0" in err

    monkeypatch.setenv("WPT_RECIPE", "d")

    def failing(step, arguments, log, last_line=False, environ=None, stops=None):
        raise Failed(step, 3)

    monkeypatch.setattr(wpt.manager, "run_command", failing)
    main([])
    assert "wpt manager stopped with exit code 3" in capsys.readouterr().err
    [run] = built.records("wpt manager")
    assert run.end["exit_code"] == 3  # the record keeps the real code

    monkeypatch.delenv("WPT_MANAGER_CONTAINER")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 3


class Priced(Fake):
    def price(self, interruptible, count=1):
        return 2.5 * count


def stopping_recipe(built, tmp_path, at_update_3):
    """A model recipe of 5 updates whose optimizer does at_update_3 (Python
    code), when the environment names STOP_AT, then waits for the manager to
    pass the stop on."""
    optimizer = tmp_path / "stopping.py"
    optimizer.write_text(
        "import os\nimport time\nimport torch\n"
        "from pathlib import Path\nfrom wpt_training import Optimizer, Verdict\n\n"
        "SETTINGS = {'lr': 0.01}\n\n"
        "class Stopping(Optimizer):\n"
        "    def __init__(self, model, settings):\n"
        "        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], "
        "lr=settings['lr'])\n\n"
        "    def before_update(self, step):\n"
        "        if step == 3 and os.environ.get('STOP_AT'):\n"
        f"            {at_update_3}\n"
        "            time.sleep(3)\n\n"
        "def start(model, settings, run):\n    return Stopping(model, settings)\n")
    stages = [{"block_size": 16, "steps": 5, "batch_size": 4}]
    return built.recipe("stopping", training={"stages": stages, "optimizer": str(optimizer)})


def checkpoint_step(built, name) -> int:
    return load_state(built.store.path(StoreCheckpoint(built.store, name, None, set()).state_file()))["step"]


@pytest.mark.parametrize("reason, ask", [("wpt manager stop", lambda: main(["stop"])),
                                         ("SIGTERM", lambda: os.kill(os.getpid(), signal.SIGTERM))])
def test_a_stop_saves_the_checkpoint_gives_the_machines_back_and_the_same_command_goes_on(
        built, fake, tmp_path, monkeypatch, reason, ask):
    # the machines go back with nothing lost, from the middle of a step, under the default WPT_LINK_CHECKPOINTS
    monkeypatch.setattr(wpt.manager, "WATCH", 0.1)
    update_3 = tmp_path / "update-3"
    monkeypatch.setenv("WPT_RECIPE", stopping_recipe(built, tmp_path, "Path(os.environ['STOP_AT']).touch()"))
    monkeypatch.setenv("STOP_AT", str(update_3))
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "wanted")

    def person():
        while not update_3.exists():
            time.sleep(0.05)
        ask()

    asking = threading.Thread(target=person, daemon=True)
    asking.start()
    with pytest.raises(SystemExit) as stopped:
        main([])
    asking.join()
    assert stopped.value.code == 130
    [(_, name, _), stop] = fake.calls
    assert stop == ("stop", name) and fake.cluster_of[name][0][0].poll() is not None
    assert checkpoint_step(built, "wanted") == 3
    assert not has_model(built.store, "wanted")
    [run] = built.records("wpt manager")
    assert [e["attributes"]["reason"] for e in run.events if e.get("name") == "manager.stop"] == [reason]
    assert not (tmp_path / "runs" / "manager-state" / "wanted.stop").exists()  # the next start is not stopped by it

    monkeypatch.delenv("STOP_AT")
    main([])
    assert has_model(built.store, "wanted")
    assert [e["attributes"]["step"] for e in built.records("wpt train")[-1].events
            if e.get("name") == "training.resumed"] == [3]


def test_wpt_manager_stop_is_refused_when_no_manager_runs_a_run(built, tmp_path, capsys):
    with pytest.raises(SystemExit) as refused:
        main(["stop"])
    assert refused.value.code == 2 and "no wpt manager runs a run" in capsys.readouterr().err


def test_the_cost_cap_prints_the_projection_first_stops_the_run_like_a_stop_and_holds_until_it_is_raised(
        built, fake, tmp_path, monkeypatch, capfd):
    monkeypatch.setattr(rental, "providers", lambda: {"fake": Priced})
    monkeypatch.setattr(wpt.manager, "WATCH", 0.1)
    reached = tmp_path / "reached"
    real = Workers.spent
    monkeypatch.setattr(Workers, "spent", lambda self, now=None: (9.0, 3.6) if reached.exists() else real(self))
    monkeypatch.setenv("WPT_RECIPE", stopping_recipe(built, tmp_path, "Path(os.environ['STOP_AT']).touch()"))
    monkeypatch.setenv("STOP_AT", str(reached))
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "capped")
    monkeypatch.setenv("WPT_BUDGET", "9")
    monkeypatch.setenv("WPT_MAX_HOURS", "100")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 130
    assert "$2.50 an hour" in capfd.readouterr().out
    [(_, name, _), stop] = fake.calls
    assert stop == ("stop", name) and checkpoint_step(built, "capped") == 3
    [run] = built.records("wpt manager")
    named = [e.get("name") for e in run.events]
    assert "manager.price" in named and named.index("manager.price") < next(
        index for index, e in enumerate(run.events) if e.get("name") == "manager.rent")
    assert [e["attributes"]["reason"] for e in run.events if e.get("name") == "manager.stop"] == [
        "the cost cap: $9.00 spent, WPT_BUDGET is $9.00"]

    # the cap is the run's, over its starts: the same command rents nothing until the cap is raised
    monkeypatch.delenv("STOP_AT")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 130 and len(fake.calls) == 2
    monkeypatch.setenv("WPT_BUDGET", "10")
    main([])
    assert has_model(built.store, "capped")


def test_a_lost_worker_is_not_replaced_once_a_stop_is_under_way(built, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "lost-stop")
    real = wpt.manager.run_command

    def losing(step, arguments, log, last_line=False, environ=None, stops=None):
        if step != "train":
            return real(step, arguments, log, last_line, environ, stops)
        (tmp_path / "runs" / "manager-state" / "lost-stop.stop").touch()
        raise Failed(step, LOST)

    monkeypatch.setattr(wpt.manager, "run_command", losing)
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == LOST
    assert [call[0] for call in fake.calls] == ["rent", "stop"]


@pytest.mark.parametrize("asked", ["cap", "stop"])
def test_a_run_that_comes_back_holding_rentals_is_stopped_during_its_commands_before_training(
        built, fake, tmp_path, monkeypatch, asked):
    # idle rentals bill during the download too: neither the cap nor a stop waits for the build
    monkeypatch.delenv("WPT_RUN", raising=False)
    monkeypatch.setattr(wpt.manager, "WATCH", 0.1)
    monkeypatch.setattr(wpt.manager, "COMMAND", ["-c", "import time; time.sleep(60)"])
    workers = cut_run(built, tmp_path)
    if asked == "cap":
        workers.state["rentals"] = [{"name": "wpt-cut-1", "price": 2.0, "started": time.time() - 7200, "stopped": None}]
        workers.save()
        monkeypatch.setenv("WPT_BUDGET", "3")
    else:
        (tmp_path / "runs" / "manager-state" / "cut.stop").touch()
    started = time.monotonic()
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 130 and time.monotonic() - started < 30
    assert ("stop", "wpt-cut-1") in fake.calls and state(tmp_path, "cut")["workers"] == []
    assert built.records("wpt train") == []


def test_no_other_run_starts_after_a_sigterm(built, fake, tmp_path, monkeypatch):
    # the container is going away: the run the settings ask for must not rent anything first
    monkeypatch.setenv("WPT_RUN", "next")
    monkeypatch.setattr(wpt.manager, "WATCH", 0.1)
    monkeypatch.setattr(wpt.manager, "COMMAND", ["-c", "import time; time.sleep(60)"])
    cut_run(built, tmp_path)
    real = wpt.manager.run_command

    def signalled(step, arguments, log, last_line=False, environ=None, stops=None):
        os.kill(os.getpid(), signal.SIGTERM)
        return real(step, arguments, log, last_line, environ, stops)

    monkeypatch.setattr(wpt.manager, "run_command", signalled)
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 130 and ("stop", "wpt-cut-1") in fake.calls
    assert not (tmp_path / "runs" / "manager-state" / "next.json").exists()


def sigterm_at_the_first_look_at_the_rentals(monkeypatch):
    """A SIGTERM before the manager claimed any run."""
    real = wpt.manager.stop_rentals

    def signalled(*arguments):
        monkeypatch.setattr(wpt.manager, "stop_rentals", real)
        os.kill(os.getpid(), signal.SIGTERM)
        return real(*arguments)

    monkeypatch.setattr(wpt.manager, "stop_rentals", signalled)


def test_a_sigterm_before_a_run_went_on_gives_its_rentals_back_and_leaves_a_run_that_holds_none(
        built, fake, tmp_path, monkeypatch):
    # the machines bill until someone gives them back, and the container is going away
    monkeypatch.delenv("WPT_RUN", raising=False)
    monkeypatch.setattr(wpt.manager, "COMMAND", ["-c", "raise SystemExit('no command may start')"])
    cut_run(built, tmp_path)
    idle = Workers.new(tmp_path / "runs" / "manager-state", "idle", "d", "test-session")
    idle.release()
    sigterm_at_the_first_look_at_the_rentals(monkeypatch)
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 130 and ("stop", "wpt-cut-1") in fake.calls
    assert state(tmp_path, "cut")["workers"] == [] and state(tmp_path, "cut")["ended"]
    assert not state(tmp_path, "idle")["ended"] and unfinished(built.store, "idle")
    assert not (tmp_path / RUN_FILE).exists()
    run = [record for record in built.records("wpt manager") if record.end is not None
           and record.start["inputs"].get("run") == "cut"][-1]
    assert run.end["exit_code"] == 130
    assert [e["attributes"]["reason"] for e in run.events if e.get("name") == "manager.stop"] == ["SIGTERM"]


def test_a_sigterm_with_no_run_to_go_on_with_starts_no_run(built, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("WPT_RUN", "next")
    sigterm_at_the_first_look_at_the_rentals(monkeypatch)
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 130 and not fake.calls and not built.records("wpt manager")


@pytest.mark.parametrize("asked, code", [(None, 0), ("next", 130)])
def test_a_sigterm_after_a_run_s_training_keeps_its_exit_code_unless_it_keeps_another_run_from_starting(
        built, fake, tmp_path, monkeypatch, asked, code):
    # a run that finished is not gone on with: 130 would have `docker compose up` start a new one
    if asked:
        monkeypatch.setenv("WPT_RUN", asked)
    else:
        monkeypatch.delenv("WPT_RUN", raising=False)
    cut_run(built, tmp_path)
    real = wpt.manager.put_model

    def signalled(*arguments):
        os.kill(os.getpid(), signal.SIGTERM)
        return real(*arguments)

    monkeypatch.setattr(wpt.manager, "put_model", signalled)
    if code:
        with pytest.raises(SystemExit) as stopped:
            main([])
        assert stopped.value.code == code
    else:
        main([])
    assert has_model(built.store, "cut")
    assert built.records("wpt manager")[-1].end["exit_code"] == 0
    assert not (tmp_path / "runs" / "manager-state" / "next.json").exists()


def test_a_command_that_outlives_the_stop_s_deadline_is_ended_and_the_machines_go_back(
        built, fake, tmp_path, monkeypatch, capfd):
    # the command ignores the stop signal, and a process it started outlives it holding its output: neither may
    # keep the machines rented
    monkeypatch.delenv("WPT_RUN", raising=False)
    monkeypatch.setenv("WPT_STOP_DEADLINE", "0.5")
    monkeypatch.setattr(wpt.manager, "WATCH", 0.1)
    ready, left = tmp_path / "ready", tmp_path / "left"
    monkeypatch.setattr(wpt.manager, "COMMAND", ["-c", f"""
import pathlib, signal, subprocess, sys, time
if 'build' in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    left = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    pathlib.Path({str(left)!r}).write_text(str(left.pid))
    pathlib.Path({str(ready)!r}).touch()
    time.sleep(60)
"""])
    cut_run(built, tmp_path)

    def person():
        while not ready.exists():
            time.sleep(0.01)
        (tmp_path / "runs" / "manager-state" / "cut.stop").touch()

    threading.Thread(target=person, daemon=True).start()
    started = time.monotonic()
    try:
        with pytest.raises(SystemExit) as stopped:
            main([])
    finally:
        if left.exists():
            os.kill(int(left.read_text()), signal.SIGKILL)
    assert stopped.value.code == 130 and time.monotonic() - started < 30
    assert ("stop", "wpt-cut-1") in fake.calls and state(tmp_path, "cut")["workers"] == []
    assert "the updates since the last checkpoint sent are lost" in capfd.readouterr().err
    run = [record for record in built.records("wpt manager") if record.start["inputs"].get("run") == "cut"][-1]
    assert [e["attributes"]["deadline"] for e in run.events if e.get("name") == "manager.stop_deadline"] == [0.5]


def test_a_stop_during_a_rental_s_wait_for_its_address_gives_the_rental_back_before_training(
        built, fake, tmp_path, monkeypatch):
    class Waiting(Fake):
        def address(self, name, watch):
            (tmp_path / "runs" / "manager-state" / "waited.stop").touch()
            while True:
                watch()
                time.sleep(0.01)

    monkeypatch.setattr(rental, "providers", lambda: {"fake": Waiting})
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "waited")
    with pytest.raises(SystemExit) as stopped:
        main([])
    assert stopped.value.code == 130
    [(_, name, _), stop] = fake.calls
    assert stop == ("stop", name) and built.records("wpt train") == []


def test_workers_rented_one_by_one_stop_being_rented_once_a_stop_comes(fake, tmp_path, monkeypatch):
    class Alone(Fake):
        cluster = False

    def watch():
        if len([call for call in fake.calls if call[0] == "rent"]) == 1:
            raise Failed("workers", 130)

    monkeypatch.setattr(rental, "providers", lambda: {"fake": Alone})
    monkeypatch.setenv("WPT_RENTAL_GPUS", "x")
    workers = Workers.new(tmp_path / "state", "pair", "d", "test-session")
    recorder = Recorder()
    with pytest.raises(Failed):
        workers.prepare({**os.environ, "WPT_GPU_WORKERS": "2"}, tmp_path, recorder, watch)
    assert [call[0] for call in fake.calls] == ["rent"]
    workers.stop(dict(os.environ), tmp_path, recorder)
    assert [call[0] for call in fake.calls] == ["rent", "stop"]


def test_skypilot_s_wait_for_an_address_ends_with_what_watch_raises(tmp_path):
    (tmp_path / "sky").write_text("#!/bin/sh\n")  # it never prints a key
    (tmp_path / "sky").chmod(0o755)
    calls = []

    def watch():
        calls.append(1)
        if len(calls) == 2:
            raise Failed("workers", 130)

    sky = SkyPilot({**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}, tmp_path)
    sky.POLL = 0.01
    with pytest.raises(Failed):
        sky.address("wpt-one", watch)


def test_a_cap_in_dollars_is_refused_without_a_price_and_a_cap_that_is_no_number_before_anything_runs(
        built, fake, monkeypatch, capsys):
    monkeypatch.setenv("WPT_GPU_WORKERS", "1")
    monkeypatch.setenv("WPT_RUN", "no-price")
    monkeypatch.setenv("WPT_BUDGET", "ten")
    with pytest.raises(SystemExit) as refused:
        main([])
    assert refused.value.code == 2 and "WPT_BUDGET='ten': a number above 0" in capsys.readouterr().err
    assert not built.records("wpt manager")

    monkeypatch.setenv("WPT_BUDGET", "10")
    with pytest.raises(SystemExit) as refused:
        main([])
    assert refused.value.code == 2 and "WPT_BUDGET needs the provider's price" in capsys.readouterr().err
    assert fake.calls == []


def test_what_a_run_spent_counts_every_rental_at_its_price_and_a_running_one_to_now(tmp_path):
    rentals = [{"name": "a", "price": 2.0, "started": 0.0, "stopped": 3600.0},
               {"name": "b", "price": 4.0, "started": 1800.0, "stopped": None},  # beside a for half an hour
               {"name": "c", "price": None, "started": 9000.0, "stopped": 10800.0}]
    workers = Workers(tmp_path / "run.json", {"run": "run", "workers": [], "rentals": rentals}, None)
    dollars, hours = workers.spent(now=10800.0)
    assert dollars == pytest.approx(2.0 + 4.0 * 2.5)  # a rental with no known price counts for the hours alone
    assert hours == pytest.approx(3.0)


def test_the_first_cap_reached_stops_the_run_and_the_projection_says_where():
    both = Cap(dollars=10.0, hours=2.0)
    assert both.reached(9.0, 1.9) is None
    assert "WPT_BUDGET" in both.reached(10.0, 0.5)
    assert "WPT_MAX_HOURS" in both.reached(1.0, 2.0)
    assert "after 2.00 more hours, at $9.00 in all" in both.projected(4.0, 1.0, 0.0)  # the dollars would last 2.25
    assert "after 0.50 more hours, at $5.00 in all" in Cap(dollars=5.0).projected(8.0, 1.0, 0.0)
    assert "no cap" in Cap().projected(8.0, 0.0, 0.0)


def test_skypilot_reads_the_price_of_the_row_it_chose_without_renting(tmp_path):
    bin_folder = tmp_path / "bin"
    bin_folder.mkdir()
    calls = tmp_path / "calls.jsonl"
    (bin_folder / "sky").write_text(f"""#!{sys.executable}
import json, sys
with open({str(calls)!r}, "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
print("Considered resources (2 nodes):")
print(" INFRA   INSTANCE     vCPUs   Mem(GB)   GPUS     COST ($)   CHOSEN")
print(" RunPod  8x_H100_SXM  128     1024      H100:8   44.32      \\x1b[32m   \\u2714\\x1b[0m")
print(" Lambda  gpu_8x_h100  208     1800      H100:8   47.60")
""", encoding="utf-8")
    (bin_folder / "sky").chmod(0o755)
    settings = {**os.environ, "PATH": f"{bin_folder}{os.pathsep}{os.environ['PATH']}", "WPT_RENTAL_GPUS": "H100:8"}
    assert SkyPilot(settings, tmp_path).price(True, 2) == 44.32
    [arguments] = [json.loads(line) for line in calls.read_text().splitlines()]
    assert arguments[:3] == ["launch", "--dryrun", "-y"] and "--use-spot" in arguments
    assert arguments[arguments.index("--num-nodes") + 1] == "2" and "-c" not in arguments

    (bin_folder / "sky").write_text(f"#!{sys.executable}\nprint('no plan')\n")
    with pytest.raises(RentalError, match="gave no price"):
        SkyPilot(settings, tmp_path).price(False)
