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

from wpt_train import LOST, main
from wpt_worker.link import Key, LinkError

ON_CPU = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}  # gloo on every system, no GPU needed


def one_stage(steps):
    return {"stages": [{"block_size": 16, "steps": steps, "batch_size": 4}]}


def start_worker(stack, tmp_path, name) -> Key:
    """A gpu worker of its own, for one manager, as a person would start
    one; its key."""
    process = subprocess.Popen([sys.executable, "-m", "wpt_worker", "--local"], stdout=subprocess.PIPE, text=True,
                               env={**os.environ, **ON_CPU, "WPT_WORKER_FOLDER": str(tmp_path / name)})
    stack.callback(process.wait, 120)
    stack.callback(process.stdout.close)
    return Key.parse(process.stdout.readline())


def train(built, tmp_path, recipe, out, workers="", *options):
    subprocess.run([sys.executable, "-m", "wpt_train", recipe, built.dataset_id, "--out", out, "--strategy", "ddp",
                    *options], cwd=tmp_path, env={**os.environ, **ON_CPU, "WPT_GPU_WORKERS": workers}, check=True)
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
        shutil.rmtree(built.store.root / "refs" / "steps")  # the hardware is no part of a step: each run trains
        with ExitStack() as stack:
            found = [start_worker(stack, tmp_path, f"{out}-{index}") for index in range(2)]
            text = " ".join(str(key).replace("#", places[index] + "#") for index, key in enumerate(found))
            two = train(built, tmp_path, recipe, out, text, "--processes", "1")
        assert one.keys() == two.keys()
        # the processes add gradients up in another order: a few bits apart
        assert all(torch.allclose(one[name], two[name], atol=1e-4) for name in one)
        run = built.records("wpt-train")[-1]
        [job] = [e for e in run.events if e["kind"] == "span_start" and e["name"] == "gpu_worker.job"]
        transports.append((job["attributes"]["workers"], job["attributes"]["transport"]))
        provenance = json.loads((tmp_path / out / "provenance.json").read_text())
        assert provenance["strategy"]["processes"] == 2
    assert transports == [(2, "nccl"), (2, "link")]


def test_a_gpu_worker_that_cannot_be_reached_or_whose_link_breaks_is_lost(built, tmp_path, monkeypatch, capsys):
    # exit code 4, recorded: what the manager holds stays, so a replacement
    # continues with --resume (wpt-manager does it for workers it rents)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    monkeypatch.setenv("WPT_GPU_WORKERS", str(Key("127.0.0.1", port, "0" * 64, "secret-token")))
    with pytest.raises(SystemExit) as lost:
        main(["d", built.dataset_id, "--out", "model"])
    assert lost.value.code == LOST
    err = capsys.readouterr().err
    assert "gpu worker lost" in err and "secret-token" not in err
    assert [e for e in built.records("wpt-train")[-1].events if e.get("name") == "gpu_worker.lost"]

    import wpt_train

    def broken(*_):
        raise LinkError("the gpu worker at 127.0.0.1:1: the other side closed the link")

    monkeypatch.delenv("WPT_GPU_WORKERS")
    monkeypatch.setattr(wpt_train, "run_jobs", broken)
    with pytest.raises(SystemExit) as lost:
        main(["d", built.dataset_id, "--out", "model"])
    assert lost.value.code == LOST and "the other side closed the link" in capsys.readouterr().err


def test_a_training_process_that_crashes_on_one_of_several_gpu_workers_ends_the_run_with_code_5_for_the_manager(
        built, tmp_path, monkeypatch, capsys):
    # the job ends on every worker, so the manager resumes from the checkpoint
    # it holds, as for a lost worker; on one worker the crash's own exit code stays
    import wpt_train

    monkeypatch.setattr(wpt_train, "run_jobs", lambda *_: 1)
    for count, expected in ((2, wpt_train.CRASHED), (1, 1)):
        with ExitStack() as stack:
            text = " ".join(str(start_worker(stack, tmp_path, f"w{count}-{index}")) for index in range(count))
            monkeypatch.setenv("WPT_GPU_WORKERS", text)
            with pytest.raises(SystemExit) as ended:
                main(["d", built.dataset_id, "--out", f"model{count}"])
        assert ended.value.code == expected
    assert wpt_train.CRASHED == 5 and wpt_train.CRASHED != LOST
    err = capsys.readouterr().err
    assert "a training process crashed (exit code 1) and the jobs on every one of the 2 gpu workers" in err
    crashes = [e for run in built.records("wpt-train") for e in run.events if e.get("name") == "gpu_worker.crashed"]
    assert [e["attributes"] for e in crashes] == [{"exit_code": 1, "workers": 2}]
