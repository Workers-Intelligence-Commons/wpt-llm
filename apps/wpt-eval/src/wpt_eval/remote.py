"""The model of a model folder, on a gpu worker (docs/machines.md,
"Evaluation"): what a test asks of the model goes through the link, and the
tests, which hold the tokenizer, the store and the sandboxes, run here.

    with Remote(folder, environ) as model:      the first gpu worker WPT_GPU_WORKERS lists, or
                                                a local one, loaded with the folder's weights
        model.config                            the model's config (wpt_model.ModelConfig)
        model.device                            where the worker runs it, in words
        model.generator(seed)                   a new seeded generator there: its number
        model.writer(prompt, rows, *, temperature, top_k, stops, routes=False, generator=None)
                                                a writer, as wpt_inference.Writer writes
        model.greedy(prompts, steps)            the tokens written greedily after each of prompts,
                                                token lists of one length

wpt_worker.evaluation.Served has the same names for a model in this
process. The worker's job (wpt_worker.evaluation) runs on its own thread
while the tests use the model; the manager sends it the checkout and the
folder's weight files, which it reads over the link, and ends it when the
block ends. Of several gpu workers, only the first is used.
"""

import shutil
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

from wpt_dataset.repo import Repo
from wpt_model import ModelConfig, weight_files
from wpt_model.checkpoint import CONFIG_NAME
from wpt_worker.link import LinkError
from wpt_worker.manager import WORKERS, Files, NoCheckout, code_root, keys, start_workers
from wpt_worker.writing import RemoteWriter, worker_generator


class WorkerError(Exception):
    """The settings name no usable gpu worker, or there is no checkout to
    send it."""


class Remote:
    def __init__(self, folder, environ, log=print):
        self.folder, self.environ, self.log = Path(folder), environ, log
        self.config = ModelConfig.load(self.folder / CONFIG_NAME)
        self.worker = self.channel = self.thread = self.files = None
        self.device = None
        self.ready, self.code, self.error = threading.Event(), None, None

    def __enter__(self):
        try:
            first = keys(self.environ)[:1]  # no key: start_workers makes a local worker
        except ValueError as error:
            raise WorkerError(str(error)) from None
        workers = start_workers({**self.environ, WORKERS: " ".join(str(key) for key in first)}, log=self.log)
        self.worker = workers[0]
        try:
            try:
                root = code_root()
            except NoCheckout as error:
                raise WorkerError(str(error)) from None
            self.worker.send_checkout(Repo(root).data_folder(), root)
            self.files = Files(Path(tempfile.mkdtemp(prefix="wpt-eval-link-")))
            weights = {name: self.files.offer(self.folder / name) for name in weight_files(self.folder)}
            spec = {"task": "wpt-eval", "model": asdict(self.config), "weights": weights}
            self.thread = threading.Thread(target=self._serve, args=(spec,), name="eval-job", daemon=True)
            self.thread.start()
            while not self.ready.wait(0.2):
                if not self.thread.is_alive():
                    raise self.error or LinkError(f"{self.worker}: the job ended with exit code {self.code} before "
                                                  "the model was loaded")
            self.log(f"device: {self.device}")
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            try:
                if self.channel is not None:
                    self.worker.call(self.channel, "eval.end")
            except LinkError:
                pass
            self.thread.join(60)
            if self.thread.is_alive():
                self.worker.stop()
                self.thread.join(60)
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self.files is not None:
            shutil.rmtree(self.files.folder, ignore_errors=True)
            self.files = None

    def generator(self, seed: int) -> int:
        return worker_generator(self.worker, self.channel, seed)

    def writer(self, prompt, rows=1, *, temperature, top_k, stops, routes=False, generator=None) -> RemoteWriter:
        return RemoteWriter(self.worker, self.channel, prompt, rows, temperature=temperature, top_k=top_k,
                            stops=stops, routes=routes, generator=generator)

    def greedy(self, prompts, steps):
        header, _ = self.worker.call(self.channel, "model.greedy", prompts=[list(row) for row in prompts],
                                     steps=steps)
        return header["tokens"]

    def _serve(self, spec):
        """The job's requests, answered until it ends."""
        handlers = {**self.files.handlers(), "ready": self._ready}
        try:
            self.code = self.worker.run(spec, handlers)
        except LinkError as error:
            self.error = error

    def _ready(self, header, body):
        self.device, self.channel = header["device"], header["channel"]
        self.ready.set()
