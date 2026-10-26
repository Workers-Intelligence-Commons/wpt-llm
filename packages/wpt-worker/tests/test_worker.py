"""The gpu worker and its link (docs/machines.md): who may reach it, what
the manager sends it, and how a job's requests and prints reach the
manager. The worker runs here on the CPU, on ports the system picks."""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import wpt_worker
from wpt_worker.link import Key, LinkError, dial, probe
from wpt_worker.manager import Worker, checkout, keys, machines

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}

# a job task of the fake checkout: it prints, asks the manager, tries the tokenizer, and ends with code 7
ECHO = '''
import json
import os

from .job import call


def run(spec, link, work):
    print("hello from the job")
    try:
        import wpt_tokenizer  # noqa: F401
        tokenizer = "imported"
    except ModuleNotFoundError as error:
        tokenizer = str(error)
    secrets = sorted(name for name in os.environ if name == "WPT_STORE" or name.startswith("AWS_"))
    header, body = call(link, "report", json.dumps({"secrets": secrets, "tokenizer": tokenizer}).encode())
    return header["code"]
'''


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, env=GIT_ENV, check=True, capture_output=True)


def fake_checkout(root: Path) -> Path:
    """A git checkout holding this wpt_worker package, its job able to run
    the task "echo" (ECHO) too."""
    package = root / "packages" / "wpt-worker" / "src" / "wpt_worker"
    shutil.copytree(Path(wpt_worker.__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    (package / "echo.py").write_text(ECHO)
    job = package / "job.py"
    job.write_text(job.read_text().replace("TASKS = {", 'TASKS = {"echo": "wpt_worker.echo", ', 1))
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "x")
    return root


@pytest.fixture
def worker():
    """A local gpu worker, started as the manager starts one, its
    environment holding store settings it must not pass on."""
    found = Worker.local_worker({**os.environ, "WPT_STORE": "file:///nowhere", "AWS_SECRET_ACCESS_KEY": "secret"})
    yield found
    found.close()


def test_a_job_runs_from_the_checkout_the_manager_sent_and_its_requests_and_prints_reach_the_manager(
        worker, tmp_path, capfd):
    # the job's process asks the manager through the worker's server and the
    # manager answers; what it prints the manager prints; its exit code
    # comes back; it holds no store settings and cannot import the tokenizer
    root = fake_checkout(tmp_path / "checkout")
    assert worker.send_checkout("data", root) > 0
    reports = []

    def report(header, body):
        reports.append(json.loads(body))
        return {"code": 7}, b""

    assert worker.run({"task": "echo"}, {"report": report}) == 7
    [found] = reports
    assert found["secrets"] == []
    assert "the gpu worker has no tokenizer" in found["tokenizer"]
    assert "hello from the job" in capfd.readouterr().out


def test_a_worker_refuses_a_link_without_its_token_and_the_manager_refuses_another_certificate(worker):
    # the token is checked before anything else, the manager sends nothing
    # to a worker that does not show the pinned certificate, and a worker
    # serves one manager at a time
    key = worker.key
    with pytest.raises(LinkError, match="another manager is connected"):
        dial(key)
    with pytest.raises(LinkError, match="wrong token"):
        dial(Key(key.host, key.port, key.pin, "not-the-token"))
    with pytest.raises(LinkError, match="not the pinned"):
        dial(Key(key.host, key.port, "0" * 64, key.token))


def test_the_checkout_sent_leaves_data_out_holds_uncommitted_changes_and_is_the_same_for_the_same_tree(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "sources").mkdir(parents=True)
    (root / "data" / "sources" / "x.json").write_text("{}")
    (root / "code.py").write_text("one\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "x")
    (root / "code.py").write_text("two\n")  # not committed
    (root / "new.py").write_text("new\n")  # not tracked
    first = checkout(root, "data")
    assert first == checkout(root, "data")

    import io
    import tarfile
    from compression import zstd

    with tarfile.open(fileobj=io.BytesIO(zstd.decompress(first))) as archive:
        names = archive.getnames()
        assert archive.extractfile("code.py").read() == b"two\n"
    assert names == ["code.py", "new.py"]


def test_a_key_says_where_its_worker_is_and_its_token_is_never_shown_whole():
    key = Key("34.0.113.5", 8470, "a" * 64, "secret-token/1", place="cluster-1", private="10.0.0.5")
    again = Key.parse(str(key))
    assert (again.host, again.port, again.pin, again.token, again.place, again.private) == (
        "34.0.113.5", 8470, "a" * 64, "secret-token/1", "cluster-1", "10.0.0.5")
    assert key.shown() == f"wpt-worker://secr********@34.0.113.5:8470?place=cluster-1&private=10.0.0.5#{'a' * 64}"
    assert Key.parse(str(Key("10.0.0.1", 1, "b" * 64, "t"))).place is None
    with pytest.raises(ValueError, match="knows only place and private"):
        Key.parse(f"wpt-worker://t@10.0.0.1:1?zone=x#{'b' * 64}")


def test_a_number_of_workers_to_rent_is_refused_where_keys_are_due():
    # wpt-manager rents the workers a number asks for; wpt-train takes keys
    with pytest.raises(ValueError, match="a number of gpu workers to rent is wpt-manager's setting"):
        keys({"WPT_GPU_WORKERS": "2"})


def test_workers_in_one_place_meet_through_nccl_and_workers_in_different_places_over_their_links():
    def placed(*found):
        return machines(list(found))

    pin = "0" * 64
    assert placed(Key("10.0.0.1", 8470, pin, "t")) == [{"index": 0, "count": 1, "transport": None}]
    # no place given: private addresses are one place (a home network), at worker 0's address, the port after its own
    home = placed(Key("192.168.1.10", 8470, pin, "t1"), Key("192.168.1.11", 8470, pin, "t2"))
    assert [(m["index"], m["count"], m["transport"], m["master"], m["port"], m["private"]) for m in home] == [
        (0, 2, "nccl", "192.168.1.10", 8471, "192.168.1.10"), (1, 2, "nccl", "192.168.1.10", 8471, "192.168.1.11")]
    # one place given by both: they meet at worker 0's private address, whatever the manager reaches it at
    cluster = placed(Key("34.0.113.1", 8470, pin, "t1", "c", "10.1.0.1"), Key("34.0.113.2", 8470, pin, "t2", "c"))
    assert {m["master"] for m in cluster} == {"10.1.0.1"} and {m["transport"] for m in cluster} == {"nccl"}
    assert [m["private"] for m in cluster] == ["10.1.0.1", "34.0.113.2"]  # worker 1's own: refused there
    # different places, a public address, or a place given by one only: over the links, each job reaching worker 0
    # at its address and pin with a secret made for the run, never worker 0's token
    first = Key("34.0.113.1", 8470, pin, "t1", "a")
    secrets = set()
    for second in (Key("34.0.113.2", 8470, pin, "t2", "b"), Key("10.0.0.2", 8470, pin, "t2")):
        across = placed(first, second)
        assert [m["transport"] for m in across] == ["link", "link"]
        hub = Key.parse(across[1]["hub"])
        assert "hub" not in across[0] and (hub.host, hub.port, hub.pin) == ("34.0.113.1", 8470, pin)
        assert hub.token == across[0]["secret"] == across[1]["secret"] and "t1" not in across[1]["hub"]
        secrets.add(hub.token)
    assert len(secrets) == 2  # one per run
    assert [m["transport"] for m in placed(Key("34.0.113.1", 8470, pin, "t1"), Key("10.0.0.2", 8470, pin, "t2"))] \
        == ["link", "link"]
    # worker 0 at this machine's own address, the others elsewhere: they would reach themselves there
    for second in (Key("192.168.1.11", 8470, pin, "t2"), Key("34.0.113.2", 8470, pin, "t2", "b")):
        with pytest.raises(ValueError, match="this machine's own"):
            placed(Key("127.0.0.1", 8470, pin, "t1", "b" if second.place else None), second)
    assert {m["master"] for m in placed(Key("127.0.0.1", 8470, pin, "t1"), Key("127.0.0.1", 8471, pin, "t2"))} \
        == {"127.0.0.1"}  # every worker on this machine


def test_nccl_peers_are_joined_at_private_addresses_only_and_only_ddp_runs_across_places():
    from wpt_worker.training import Refused, interface_of, placement

    options = {"processes": 2, "strategy": "ddp"}
    # this machine's own address stands for a private one it has; NCCL and gloo keep to its interface
    near = {"index": 1, "count": 2, "transport": "nccl", "master": "192.168.1.10", "port": 8471,
            "private": "127.0.0.1"}
    interface = interface_of("127.0.0.1")
    assert interface is not None
    assert placement(near, options) == {
        "environ": {"NCCL_SOCKET_IFNAME": f"={interface}", "GLOO_SOCKET_IFNAME": interface}, "first_rank": 2,
        "world_size": 4, "master": ("192.168.1.10", 8471)}
    for wrong in ({"master": "8.8.8.8"}, {"private": "8.8.8.8"}):  # worker 0's, or this worker's own
        with pytest.raises(Refused, match="not a private address"):
            placement({**near, **wrong}, options)
    with pytest.raises(Refused, match="none of this machine's network interfaces has"):
        placement({**near, "private": "10.255.254.253"}, options)
    far = {"index": 1, "count": 2, "transport": "link", "hub": "wpt-worker://..."}
    found = placement(far, options)
    assert found["first_rank"] == 2 and found["master"] is None
    assert found["environ"] == {"WPT_MACHINE": "1", "WPT_MACHINES": "2", "WPT_BRIDGED": "1",
                                "WPT_PEER_HUB": "wpt-worker://..."}
    with pytest.raises(Refused, match="use --strategy ddp"):
        placement(far, {**options, "strategy": "split"})


def test_a_worker_is_there_while_another_manager_holds_it_and_not_once_it_is_gone(worker):
    assert probe(worker.key)
    assert not probe(Key(worker.key.host, worker.key.port, "0" * 64, worker.key.token))  # another certificate
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    assert not probe(Key("127.0.0.1", port, worker.key.pin, worker.key.token), timeout=2)


def test_a_peer_needs_the_secret_of_the_run_whose_job_runs_never_the_worker_s_token(worker, tmp_path):
    key = worker.key

    def as_peer(token, **options):
        return dial(Key(key.host, key.port, key.pin, token), peer=1, **options)

    with pytest.raises(LinkError, match="no job of that run runs on this gpu worker"):
        as_peer("run-secret")  # no job yet
    root = fake_checkout(tmp_path / "checkout")
    worker.send_checkout("data", root)
    checked = []

    def report(header, body):  # while the job runs
        for token in (key.token, "not-the-secret"):
            with pytest.raises(LinkError, match="no job of that run runs on this gpu worker"):
                as_peer(token)
        with pytest.raises(LinkError, match="did not answer"):  # kept for the job, which never takes it
            as_peer("run-secret", answer=1)
        checked.append(True)
        return {"code": 0}, b""

    assert worker.run({"task": "echo", "machines": {"secret": "run-secret"}}, {"report": report}) == 0
    assert checked


def test_a_token_set_for_a_worker_is_never_printed_whole_and_a_worker_with_no_manager_ends(tmp_path):
    # the manager sets the token of a worker it rents; WPT_WORKER_WAIT
    # (seconds) is how long the worker waits with no manager, then it ends
    from wpt_worker.server import Settings

    assert Settings({}).wait == Settings({"WPT_WORKER_WAIT": ""}).wait == 3600  # 1 hour, unless set
    token = "manager-set-token-0123456789"
    started = subprocess.run([sys.executable, "-m", "wpt_worker"], capture_output=True, text=True, timeout=120,
                             env={**os.environ, "WPT_WORKER_TOKEN": token, "WPT_WORKER_PORT": "0",
                                  "WPT_WORKER_ADDRESS": "127.0.0.1", "WPT_WORKER_WAIT": "1",
                                  "WPT_WORKER_FOLDER": str(tmp_path / "worker")})
    assert started.returncode == 0
    assert token not in started.stdout + started.stderr
    shown = Key.parse(started.stdout.splitlines()[0])
    assert shown.token == "mana********" and "no manager for 1 s" in started.stdout


def test_a_job_s_first_start_removes_the_checkpoints_earlier_runs_left_on_the_worker(tmp_path):
    # each job's checkpoints are keyed by its run's record folder, which no
    # later command uses again: after a stop or a crash, nothing reads them
    from wpt_worker.training import _remove_others

    for name in ("earlier-1", "earlier-2", "this-run"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "checkpoint.safetensors").write_bytes(b"state")
    said = []
    _remove_others(tmp_path / "this-run", said.append)
    assert [path.name for path in tmp_path.iterdir()] == ["this-run"] and len(said) == 2


def test_a_value_broadcast_between_machines_crosses_as_json_and_tensor_bytes_and_arrives_the_same():
    # a checkpoint's state: tensors of several types, numbered keys, tuples;
    # nothing pickled crosses, and a type JSON cannot say is refused
    import threading

    import torch

    from wpt_worker.peers import Exchange, packed

    value = {"model": {"w": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), "none": torch.empty(0, 4)},
             "optimizer": {"state": {0: {"step": torch.tensor(3.0)}}, "param_groups": [{"betas": (0.9, 0.95),
                                                                                         "params": [0]}]},
             "flags": [True, None, 1, 2.5, "text"], "big": torch.ones(20 << 20, dtype=torch.uint8)}  # more than one part
    hub_side, spoke_side = socket.socketpair()
    with hub_side, spoke_side:
        sent = threading.Thread(target=Exchange(0, 2, [hub_side]).broadcast, args=(value,))
        sent.start()
        got = Exchange(1, 2, [spoke_side]).broadcast(None)
        sent.join()

    def same(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if isinstance(a, dict):
            return type(b) is dict and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return type(a) is type(b) and a == b

    assert same(value, got)
    with pytest.raises(TypeError, match="cannot cross between gpu workers"):
        packed({"a": {1, 2}}, [])


def test_every_link_notices_a_machine_that_is_gone_without_closing_it():
    from wpt_worker.link import keep_alive

    with socket.socket() as sock:
        keep_alive(sock)
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        if hasattr(socket, "TCP_KEEPIDLE"):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 30


def test_a_worker_prints_the_place_and_private_address_its_settings_give(tmp_path):
    # a machine of a cluster the manager rented in one request: start.sh sets both
    started = subprocess.run([sys.executable, "-m", "wpt_worker"], capture_output=True, text=True, timeout=120,
                             env={**os.environ, "WPT_WORKER_PORT": "0", "WPT_WORKER_ADDRESS": "127.0.0.1",
                                  "WPT_WORKER_WAIT": "1", "WPT_WORKER_PLACE": "wpt-run-1a2b3c",
                                  "WPT_WORKER_PRIVATE": "10.0.0.7", "WPT_WORKER_FOLDER": str(tmp_path / "worker")})
    key = Key.parse(started.stdout.splitlines()[0])
    assert (key.place, key.private) == ("wpt-run-1a2b3c", "10.0.0.7")
    plain = subprocess.run([sys.executable, "-m", "wpt_worker"], capture_output=True, text=True, timeout=120,
                           env={**{n: v for n, v in os.environ.items() if not n.startswith("WPT_WORKER_")},
                                "WPT_WORKER_PORT": "0", "WPT_WORKER_ADDRESS": "127.0.0.1", "WPT_WORKER_WAIT": "1",
                                "WPT_WORKER_FOLDER": str(tmp_path / "plain")})
    assert (Key.parse(plain.stdout.splitlines()[0]).place, Key.parse(plain.stdout.splitlines()[0]).private) == (None, None)
