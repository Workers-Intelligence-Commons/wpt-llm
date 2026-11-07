"""A run on several gpu workers, and a lost gpu worker (docs/machines.md,
"Several machines" and "A lost gpu worker"). The workers run here on the
CPU, each a process of its own on a port the system picks."""

import json
import os
import shutil
import socket
import subprocess
import sys
from contextlib import ExitStack

import pytest
import torch
from safetensors.torch import load_file

from wpt.train import LOST, main
from wpt_training import load_state
from wpt_utils.lines import bare
from wpt_worker.link import Key, LinkError

pytestmark = pytest.mark.usefixtures("on_the_cpu")  # gloo on every system, no GPU needed


def one_stage(steps):
    return {"stages": [{"block_size": 16, "steps": steps, "batch_size": 4}]}


def start_worker(stack, tmp_path, name) -> Key:
    """A gpu worker of its own, for one manager, as a person would start
    one; its key."""
    process = subprocess.Popen([sys.executable, "-m", "wpt_worker", "--local"], stdout=subprocess.PIPE, text=True,
                               env={**os.environ, "WPT_WORKER_FOLDER": str(tmp_path / name)})
    stack.callback(process.wait, 120)
    stack.callback(process.stdout.close)
    return Key.parse(bare(process.stdout.readline()))


def train(built, tmp_path, recipe, out, workers="", *options):
    subprocess.run([sys.executable, "-m", "wpt", "train", recipe, built.dataset_id, "--out", out, "--strategy", "ddp",
                    *options], cwd=tmp_path, env={**os.environ, "WPT_GPU_WORKERS": workers}, check=True)
    return load_file(tmp_path / out / "model.safetensors")


def test_two_gpu_workers_train_the_same_model_as_two_processes_on_one_in_one_place_and_across_places(built,
                                                                                                    tmp_path):
    # in one place (private addresses, no place given), the workers' processes
    # meet through gloo here (NCCL on GPUs); across places (each key names
    # another place), they exchange gradients over their links: either way
    # the same synced training as two processes of one worker
    recipe = built.recipe("four", training=one_stage(4))
    one = train(built, tmp_path, recipe, "one", "", "--processes", "2")  # one local worker
    transports = []
    for out, places in (("near", ("", "")), ("far", ("?place=a", "?place=b"))):
        shutil.rmtree(built.store.root / "cache" / "steps")  # the hardware is no part of a step: each run trains
        with ExitStack() as stack:
            found = [start_worker(stack, tmp_path, f"{out}-{index}") for index in range(2)]
            text = " ".join(str(key).replace("#", places[index] + "#") for index, key in enumerate(found))
            two = train(built, tmp_path, recipe, out, text, "--processes", "1")
        assert one.keys() == two.keys()
        # the processes add gradients up in another order: a few bits apart
        assert all(torch.allclose(one[name], two[name], atol=1e-4) for name in one)
        run = built.records("wpt train")[-1]
        [job] = [e for e in run.events if e["kind"] == "span_start" and e["name"] == "gpu_worker.job"]
        transports.append((job["attributes"]["workers"], job["attributes"]["transport"]))
        provenance = json.loads((tmp_path / out / "provenance.json").read_text())
        assert provenance["strategy"]["processes"] == 2
    assert transports == [(2, "nccl"), (2, "link")]


def test_a_gpu_worker_that_cannot_be_reached_or_whose_link_breaks_is_lost(built, tmp_path, monkeypatch, capsys):
    # exit code 4, recorded: what the manager holds stays, so a replacement
    # continues with --resume (wpt manager does it for workers it rents)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    monkeypatch.setenv("WPT_GPU_WORKERS", str(Key("127.0.0.1", port, "0" * 64, "secret-token")))
    with pytest.raises(SystemExit) as lost:
        main(["d", built.dataset_id, "--out", "model"])
    assert lost.value.code == LOST
    err = capsys.readouterr().err
    assert "gpu worker lost" in err and "secret-token" not in err
    assert [e for e in built.records("wpt train")[-1].events if e.get("name") == "gpu_worker.lost"]

    import wpt.train

    def broken(*_):
        raise LinkError("the gpu worker at 127.0.0.1:1: the other side closed the link")

    monkeypatch.delenv("WPT_GPU_WORKERS")
    monkeypatch.setattr(wpt.train, "run_jobs", broken)
    with pytest.raises(SystemExit) as lost:
        main(["d", built.dataset_id, "--out", "model"])
    assert lost.value.code == LOST and "the other side closed the link" in capsys.readouterr().err


def test_a_training_process_that_crashes_on_one_of_several_gpu_workers_ends_the_run_with_code_5_for_the_manager(
        built, tmp_path, monkeypatch, capsys):
    # the job ends on every worker, so the manager resumes from the checkpoint
    # it holds, as for a lost worker; on one worker the crash's own exit code stays
    import wpt.train

    monkeypatch.setattr(wpt.train, "run_jobs", lambda *_: 1)
    for count, expected in ((2, wpt.train.CRASHED), (1, 1)):
        with ExitStack() as stack:
            text = " ".join(str(start_worker(stack, tmp_path, f"w{count}-{index}")) for index in range(count))
            monkeypatch.setenv("WPT_GPU_WORKERS", text)
            with pytest.raises(SystemExit) as ended:
                main(["d", built.dataset_id, "--out", f"model{count}"])
        assert ended.value.code == expected
    assert wpt.train.CRASHED == 5 and wpt.train.CRASHED != LOST
    err = capsys.readouterr().err
    assert "a training process crashed (exit code 1) and the jobs on every one of the 2 gpu workers" in err
    crashes = [e for run in built.records("wpt train") for e in run.events if e.get("name") == "gpu_worker.crashed"]
    assert [e["attributes"] for e in crashes] == [{"exit_code": 1, "workers": 2}]


def patched_recipe(built, tmp_path, name, steps, patch):
    """A recipe whose optimizer file runs patch (Python code, with job the
    module wpt_worker.training) in each training process that has STOP set."""
    optimizer = tmp_path / f"{name}.py"
    optimizer.write_text(
        "import os\nimport signal\nimport time\nimport torch\nfrom wpt_training import Optimizer\n\n"
        "SETTINGS = {'lr': 0.01}\n\n"
        "class Plain(Optimizer):\n"
        "    def __init__(self, model, settings):\n"
        "        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], "
        "lr=settings['lr'])\n\n"
        "def start(model, settings, run):\n"
        "    if os.environ.get('STOP'):\n"
        "        import wpt_worker.training as job\n"
        + "".join(f"        {line}\n" for line in patch.splitlines())
        + "    return Plain(model, settings)\n")
    stages = [{"block_size": 16, "steps": steps, "batch_size": 4}]
    return built.recipe(name, training={"stages": stages, "optimizer": str(optimizer)})


def started(built, tmp_path, recipe, *options, **environ):
    return subprocess.run([sys.executable, "-m", "wpt", "train", recipe, built.dataset_id, "--out", "model",
                           "--strategy", "ddp", "--processes", "2", *options], cwd=tmp_path, capture_output=True,
                          text=True, env={**os.environ, **environ})


def test_a_limit_on_two_processes_waits_for_the_main_one_to_send_its_checkpoint(built, tmp_path):
    # the other process is done at once; the checkpoint that takes its time to send must still arrive.
    # gloo's closing waits for every process, which NCCL's need not, so here closing does not wait
    patch = ("torch.distributed.destroy_process_group = lambda: None\n"
             "send = job.LinkCheckpoint.send\n"
             "job.LinkCheckpoint.send = lambda self: (time.sleep(3), send(self))[1]")
    recipe = patched_recipe(built, tmp_path, "limited", 5, patch)
    done = started(built, tmp_path, recipe, "--stop-after-updates", "3", STOP="1")
    assert done.returncode == 130, done.stderr
    assert "measured 2 updates in " in done.stdout
    assert load_state(tmp_path / "model" / "checkpoint" / "checkpoint.safetensors")["step"] == 3


def test_a_stop_during_a_step_s_end_keeps_its_result_stops_every_process_and_the_run_goes_on(built, tmp_path):
    # the signal comes while the main process sends the step's result: a whole step of work
    patch = ("upload = job.upload\n"
             "def stopped(link, path):\n"
             "    os.kill(os.getpid(), signal.SIGTERM)\n"
             "    time.sleep(0.2)\n"
             "    return upload(link, path)\n"
             "job.upload = stopped")
    recipe = patched_recipe(built, tmp_path, "ending", 2, patch)
    done = started(built, tmp_path, recipe, "--max-restarts", "0", STOP="1")
    assert done.returncode == 130, done.stderr
    assert not (tmp_path / "model" / "config.json").exists()

    done = started(built, tmp_path, recipe, "--max-restarts", "0", "--resume")
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "model" / "config.json").is_file()
    events = built.records("wpt train")[-1].events
    assert [e for e in events if e.get("name") == "training.step_reused"]
    assert not [e for e in events if e.get("name") == "training.step"]
