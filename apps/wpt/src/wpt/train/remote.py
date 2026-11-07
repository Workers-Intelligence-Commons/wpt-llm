"""wpt train's side of the links to its gpu workers (docs/machines.md):
what the workers' training jobs (wpt_worker.training) ask, answered from
the manager's store, record, checkpoint and tokenizer. Each worker's link
is served on a thread of its own (wpt_worker.manager.run_jobs); the
answers are the same for every worker, and only the run's main process
(on worker 0) asks ready, checkpoint.*, step.done and finished.

    ready             the job's device, processes, libraries' versions and pass-time setup; answered
                      with the plan: each step's fingerprint (on this side's code,
                      wpt_utils.programs.code: the checkout the worker runs), the steps reused, the state the
                      run starts from, the scheduler's pass times to replay
    file.offer        a file of the run's dataset or of a state it starts from
    record.*          a file of the run's record folder (read, write, append), and only of it;
                      flush: the lines the manager's store holds back, sent
    checkpoint.*      the run's checkpoint: exists, load, save (a file the job sent), remove
    env.call          a method of a step's environment part on the manager (the part made by
                      the step's first call from a process of a worker, dropped when the
                      step ends)
    step.done         a step ended: its result kept (the checkpoint's state, or the state the
                      job sent with it), the checkpoint removed
    finished          the drafter and the sample's token ids, once every step is done
    file.range, upload.*   the files offered and sent (wpt_worker.manager.Files)

wpt_worker.writing.RemoteWriter is the writer an environment's manager
part gets (Context.writer): the model on the gpu worker writes its tokens.
"""

import functools
import json
import threading
from dataclasses import asdict

from wpt_inference import replayed_costs
from wpt_store import File
from wpt_training import Context
from wpt_utils.programs import code
from wpt_worker.writing import RemoteWriter

def named_files(value) -> dict[str, File]:
    """{sha256: File} of every file a dataset record refers to: each
    reference {"key", "sha256"} in it, shards included."""
    if isinstance(value, dict):
        if set(value) >= {"key", "sha256"}:
            return {value["sha256"]: File.of(value)}
        return {sha: file for part in value.values() for sha, file in named_files(part).items()}
    if isinstance(value, list):
        return {sha: file for part in value for sha, file in named_files(part).items()}
    return {}


class Training:
    """The answers to one run's jobs; see the module's docstring. run: what
    the caller knows of the run (wpt.train._train), workers: its gpu
    workers, refuse: the exception a refusal raises (its type's name is
    what the job hears)."""

    def __init__(self, run, workers, files, refuse):
        self.run, self.workers, self.files, self.refuse = run, workers, files, refuse
        self.allowed = named_files(run.record)  # the files the gpu worker may ask for, by their sha256
        self.parts = {}  # (worker's number, channel, step number): the environment's part on the manager
        self.lock = threading.Lock()  # around self.parts, which every worker's thread changes
        self.plan = self.ready_found = self.finished = None
        self.done, self.outputs = list(run.done), {}
        self.offered = []  # the files of the store copied to this machine for the job, released at the end

    def handlers(self, worker) -> dict:
        """The answers to worker's job."""
        return {"ready": self._ready, "file.offer": self._file, "record.read": self._read,
                "record.write": self._write, "record.append": self._append, "record.flush": self._flush,
                "checkpoint.exists": self._exists,
                "checkpoint.load": self._load, "checkpoint.save": self._save, "checkpoint.remove": self._remove,
                "env.call": functools.partial(self._environment, worker), "step.done": self._step_done,
                "finished": self._finished, **self.files.handlers()}

    def close(self):
        for file in self.offered:
            self.run.store.release(file)
        self.files.close()

    def _offer(self, file: File) -> dict:
        sha = file.sha256
        path = self.run.store.path(file)  # checked against its sha256
        self.offered.append(file)
        return self.files.offer(path, sha)

    def _ready(self, header, body):
        run, telemetry = self.run, self.run.telemetry
        self.ready_found = {"device": header["device"], "processes": header["processes"], "tools": header["tools"]}
        costs = None
        if run.args.scheduler_costs:
            try:
                costs = replayed_costs(run.replayed, header["setup"])
            except ValueError as error:
                raise self.refuse(f"ValueError: {error}") from None
        run.setup, run.parameters = header["setup"], header["parameters"]
        processes = header["processes"]
        run.log(f"device: {header['device']}" + (f", {processes} processes" if processes > 1 else ""))
        marks = run.fingerprints(code())  # the checkout this side sent the worker
        telemetry.event("training.settings", recipe=run.recipe, model=asdict(run.model_config),
                        parameters=run.parameters, device=header["device"], processes=processes,
                        strategy={"name": run.strategy_file.name, "file_sha256": run.strategy_file.sha256},
                        steps=[{**step.record(), "fingerprint": mark} for step, mark in zip(run.steps, marks)],
                        resume=run.args.resume, checkpoint=str(run.checkpoint))
        kept = [run.results.found(mark) for mark in marks]
        reused = max((number for number, result in enumerate(kept, 1) if result), default=0)
        state = None
        if reused:
            run.kept_states.update(result["state"]["key"] for result in kept if result)
            found = kept[:reused]
            self.done = [{**result["about"], "result": result["result"], "reused": True} for result in found]
            telemetry.input(reused_steps={mark: result["result"] for mark, result in zip(marks, found)})
            for step, about in zip(run.steps, self.done):
                telemetry.event("training.step_reused", step=step.name, number=about["number"],
                                fingerprint=about["fingerprint"], result=about["result"])
                run.log(f"step {about['number']} {step.name}: reused, result {about['result'][:12]}")
            state = File.of(found[-1]["state"])
            self.allowed[state.sha256] = state
        self.marks, self.last_state = marks, state
        self.plan = {"marks": marks, "reused": reused, "state": None if state is None else self._offer(state),
                     "costs": costs}
        return self.plan, b""

    def _file(self, header, body):
        sha = header["sha"]
        if sha not in self.allowed:
            raise ValueError(f"file {sha} is not one of this run's: the gpu worker reads only the dataset's "
                             "files and the states the run starts from")
        return self._offer(self.allowed[sha]), b""

    def _folder(self, header):
        if header["folder"] != self.run.telemetry.folder:
            raise ValueError(f"{header['folder']} is not this run's record folder")
        return header["folder"]

    def _read(self, header, body):
        return {}, self.run.store.read_record(self._folder(header), header["name"])

    def _write(self, header, body):
        self.run.store.write_record(self._folder(header), header["name"], body)

    def _append(self, header, body):
        self.run.store.append_record(self._folder(header), header["name"], body)

    def _flush(self, header, body):
        self.run.store.flush_records()

    def _exists(self, header, body):
        return {"exists": self.run.checkpoint.exists()}, b""

    def _load(self, header, body):
        state, path = self.run.checkpoint_state(), self.run.checkpoint_file()
        if state is not None:
            self.allowed[state["sha256"]] = File.of(state)
            return self._offer(File.of(state)), b""
        if path is not None:
            return self.files.offer(path), b""
        return {"file": None}, b""

    def _save(self, header, body):
        path = self.files.received(header["upload"])
        try:
            self.run.checkpoint.save_file(path)
        finally:
            path.unlink(missing_ok=True)

    def _remove(self, header, body):
        self.run.checkpoint.remove()

    def _environment(self, worker, header, body):
        asked = json.loads(body)
        number, name = asked["number"], asked["name"]
        if name.startswith("_"):
            raise ValueError(f"{name}: not a method an environment's part offers")
        key = (self.workers.index(worker), header.get("channel"), number)
        with self.lock:
            part = self.parts.get(key)
        if part is None:
            run, step = self.run, self.run.steps[number - 1]
            context = Context(step.name, run.store, run.record, run.tokenizer, run.model_config, run.telemetry,
                              run.log, run.recipe["seed"], number,
                              writer=functools.partial(RemoteWriter, worker, header.get("channel")))
            part = step.start_manager(context)
            if part is None:
                raise ValueError(f"the environment of step {number} {step.name} has no part on the manager")
            with self.lock:
                # a process that starts the step again (after a crash) gets a new part, on a new channel:
                # the parts of the worker's earlier channels go
                self.parts = {found: value for found, value in self.parts.items()
                              if found[0] != key[0] or found[1] == key[1]}
                self.parts[key] = part
        return {}, json.dumps(getattr(part, name)(*asked["arguments"])).encode()

    def _step_done(self, header, body):
        run, about = self.run, json.loads(body)
        number = about["number"]
        step, mark = run.steps[number - 1], self.marks[number - 1]
        kept = {"number": number, **step.record(), "fingerprint": mark, **about["reached"],
                "sources": about["sources"]}
        with run.telemetry.span("training.step_kept", step=step.name, number=number, fingerprint=mark) as span:
            if header.get("state") is not None:  # the job sent the step's last checkpoint with its end
                path = self.files.received(header["state"])
                try:
                    state = run.store.file(run.store.put_file(path)).reference()
                finally:
                    path.unlink(missing_ok=True)
            else:
                state = run.kept_state()
            sha = run.results.keep(mark, state, kept)
            run.kept_states.add(state["key"])  # the checkpoint's record goes, its file stays as the step's state
            span.set(result=sha)
        self.outputs[mark] = sha
        run.telemetry.output(step_results=self.outputs)
        self.done.append({**kept, "result": sha, "reused": False})
        self.last_state = state
        run.checkpoint.remove()  # the step's result is kept
        with self.lock:
            self.parts = {key: part for key, part in self.parts.items() if key[2] != number}
        return {"result": sha}, b""

    def _finished(self, header, body):
        self.finished = {**json.loads(body), "drafter": self.files.received(header["drafter"])}

