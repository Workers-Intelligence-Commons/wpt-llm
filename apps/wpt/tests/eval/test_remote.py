"""The model of an evaluation on a gpu worker (docs/machines.md,
"Evaluation"): the worker runs here, on the CPU, as a process of its own
that the manager's side reaches over the link."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from wpt.eval import main
from wpt.eval.remote import Remote
from wpt_model import load_model
from wpt_tokenizer import END
from wpt_utils.lines import bare
from wpt_worker.evaluation import Served
from wpt_worker.link import Key

pytestmark = pytest.mark.usefixtures("on_the_cpu")
PROMPTS = [[END, 5, 6, 7, 8], [END, 9, 10, 11, 12], [END, 13, 14, 15, 16]]


def test_the_model_on_a_gpu_worker_writes_what_the_same_model_in_this_process_writes(trained):
    # the weights cross the link as files, the worker loads them, and what
    # the tests ask (greedy tokens, a writer on a seeded generator) comes back as tokens
    local = Served(load_model(trained.folder))
    with Remote(trained.folder, dict(os.environ)) as remote:
        assert remote.config == local.config and remote.device == local.device
        assert remote.greedy(PROMPTS, 6) == local.greedy(PROMPTS, 6)
        windows = [[END, *range(5, 30)], [END, 7]]  # the held-out-loss test's windows cross the link as token ids
        for far, near in zip(remote.losses(windows), local.losses(windows)):
            np.testing.assert_allclose(far, near, rtol=1e-6)
        settings = {"temperature": 0.8, "top_k": 40, "stops": [END]}
        far = remote.writer(PROMPTS[0], 2, generator=remote.generator(7), **settings)
        near = local.writer(PROMPTS[0], 2, generator=local.generator(7), **settings)
        for _ in range(3):  # the generator goes on from where it was, one writer after the other
            assert far.write({}, {0: 5, 1: 5}) == near.write({}, {0: 5, 1: 5})
        far = remote.writer(PROMPTS[1], 1, generator=remote.generator(7), **settings)
        near = local.writer(PROMPTS[1], 1, generator=local.generator(7), **settings)
        assert far.write({}, {0: 8}) == near.write({}, {0: 8})
        assert remote.worker is not None
    assert remote.worker is None  # the job ended and the link closed with the block


def test_a_writer_stops_at_a_stop_token_and_at_its_limit(trained):
    with Remote(trained.folder, dict(os.environ)) as remote:
        [free] = remote.greedy([PROMPTS[0]], 6)
        stop = free[2]
        written = remote.writer(PROMPTS[0], 1, temperature=1.0, top_k=1, stops=[stop]).write({}, {0: 6})
        assert written == {0: free[:free.index(stop) + 1]}
        assert remote.writer(PROMPTS[0], 1, temperature=1.0, top_k=1, stops=[]).write({}, {0: 3}) == {0: free[:3]}


def start_worker(stack, tmp_path, name) -> Key:
    process = subprocess.Popen([sys.executable, "-m", "wpt_worker", "--local"], stdout=subprocess.PIPE, text=True,
                               env={**os.environ, "WPT_WORKER_FOLDER": str(tmp_path / name)})
    stack.callback(process.wait, 120)
    stack.callback(process.stdout.close)
    return Key.parse(bare(process.stdout.readline()))


def test_the_command_uses_the_first_gpu_worker_listed_and_reaches_no_other(trained, tmp_path, monkeypatch):
    from contextlib import ExitStack

    test = tmp_path / "where.py"
    test.write_text("from wpt.eval import Result\n\ndef run(context):\n"
                    "    return Result({'device': context.device}, 'ran on ' + context.device)\n")
    with ExitStack() as stack:
        key = start_worker(stack, tmp_path, "first")
        nobody = Key("127.0.0.1", 1, "0" * 64, "nobody")  # not running: reaching it would fail the run
        monkeypatch.setenv("WPT_GPU_WORKERS", f"{key} {nobody}")
        main([str(trained.folder), str(test)])
    result = json.loads((trained.folder / "evals" / "where.json").read_text())
    assert result["data"]["device"] == result["device"] and result["device"]


def test_a_number_of_gpu_workers_to_rent_is_refused_before_anything_runs(trained, tmp_path, monkeypatch):
    monkeypatch.setenv("WPT_GPU_WORKERS", "2")
    with pytest.raises(SystemExit, match="WorkerError: WPT_GPU_WORKERS=2: a number of gpu workers to rent"):
        main([str(trained.folder)])
    assert not (trained.folder / "evals").exists()


def test_kernels_named_that_cannot_work_on_the_gpu_worker_refuse_the_evaluation_before_it_runs(trained, tmp_path,
                                                                                             monkeypatch, capsys):
    # docs/model.md, "Attention": a named form is the one that runs or the
    # run stops with the reason; here a worker with no GPU asked for Triton's
    # kernels, before the model is loaded and before any test. A name that is
    # no form is refused too
    test = tmp_path / "ran.py"
    test.write_text("from wpt.eval import Result\n\ndef run(context):\n    return Result({}, 'ran')\n")
    with pytest.raises(SystemExit) as refused:
        main([str(trained.folder), str(test), "--kernels", "triton"])
    assert refused.value.code != 0
    captured = capsys.readouterr()
    assert "KernelUnavailable: Triton kernels run on a GPU, and this process runs on cpu" in captured.out + captured.err
    assert not (trained.folder / "evals" / "ran.json").exists()

    monkeypatch.setenv("WPT_KERNELS", "fast")
    with pytest.raises(SystemExit) as refused:
        main([str(trained.folder), str(test)])
    assert refused.value.code == 2 and "WPT_KERNELS='fast': not one of tilelang, triton, pytorch" in \
        capsys.readouterr().err
    assert not (trained.folder / "evals" / "ran.json").exists()


def test_the_kernels_named_pytorch_are_carried_to_the_job_and_the_model_still_writes(trained):
    with Remote(trained.folder, dict(os.environ), kernels="pytorch") as remote:
        assert remote.greedy(PROMPTS, 3) == Served(load_model(trained.folder)).greedy(PROMPTS, 3)
