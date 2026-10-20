"""Shared setup for the tests of every package and application: one
PyTorch thread per test process when pytest-xdist runs the tests in
parallel (scripts/check.py), whether Docker runs on this machine, and how
two passes over the same tokens may differ in routing. Test files reach the
last two as fixtures (they cannot import this file under
--import-mode=importlib)."""

import os
import shutil
import subprocess

import pytest

# In a pytest-xdist worker, PyTorch and the programs it starts use 1 thread:
# every worker taking every core makes the machine unusable. The threading
# libraries read these variables when torch is first imported, and
# apps/conftest.py imports it before pytest_configure runs, so they are set
# here, when pytest loads this file; a serial run keeps the default.
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"


def pytest_configure(config):
    """In a pytest-xdist worker, also limits torch's own thread pool to 1,
    which applies even when torch was imported before this hook."""
    if "PYTEST_XDIST_WORKER" not in os.environ:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def docker_runs():
    """Skips the test, with the reason, when Docker is not installed or its
    daemon does not answer."""
    docker = shutil.which("docker")
    if docker is None or subprocess.run([docker, "info"], capture_output=True).returncode:
        pytest.skip("Docker is not installed or its daemon does not answer")


@pytest.fixture(scope="session")
def routing():
    """How two passes over the same tokens may differ in routing (docs/training.md,
    "Comparing routing in tests"): ROUNDING, how far the passes may put a router
    score (a sigmoid in (0, 1), in FP32), the FP32 rounding of every layer before
    the router added up; ARITHMETIC, what a deciding comparison's own arithmetic
    may round, 4 units of FP32 rounding; first_difference(model, text, routes)."""
    import torch
    from wpt_model.experts import Experts, Routes

    class Routing:
        ROUNDING = 2.0 ** -14
        ARITHMETIC = 4 * torch.finfo(torch.float32).eps

        @classmethod
        def first_difference(cls, model, text, routes):
            """Reads text (token ids of one sequence) whole in model, recording its
            routes, and compares them with routes (a Routes that covers the
            text, such as a cache's): the first position where the routed
            experts differ, None when none does. A position may differ only at
            a near tie, where the whole pass's k-th and (k+1)-th biased scores
            lie no farther apart than two scores each moved by ROUNDING, plus
            ARITHMETIC; the caller compares outputs only before that position."""
            gaps, hooks = {}, []
            for experts in (m for m in model.modules() if isinstance(m, Experts) and not m.hashed):
                def gap(_, __, logits, experts=experts):
                    top = torch.topk(torch.sigmoid(logits.float()) + experts.bias, experts.k + 1).values
                    gaps[experts] = top[:, experts.k - 1] - top[:, experts.k]
                hooks.append(experts.router.register_forward_hook(gap))
            recorded = Routes()
            try:
                with torch.no_grad():
                    model(torch.tensor([text]), record=recorded)
            finally:
                for hook in hooks:
                    hook.remove()
            first = None
            for experts, chosen in recorded.chosen.items():
                known = routes.replay(experts, len(text))
                differ = (chosen.sort(-1).values != known.sort(-1).values).any(-1)
                assert (gaps[experts][differ] <= 2 * cls.ROUNDING + cls.ARITHMETIC).all(), \
                    "a token whose routing is clear went to other experts"
                if differ.any():
                    position = int(differ.nonzero()[0])
                    first = position if first is None else min(first, position)
            return first

    return Routing
