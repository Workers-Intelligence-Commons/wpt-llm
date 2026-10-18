"""Model drafts for the conversation editor: a WPT model folder on this
machine writes the next assistant message of a conversation, as `wpt-chat
--chat` does (the chat template's prompt, sampled until <|im_end|>). The
person then edits the draft or confirms it as it is; the editor saves it
with the metadata.draft this module returns, whose `edited` the service
layer sets at save against the drafted content's sha256
(docs/chat-format.md, "Model drafts"). The server checks nothing else of
that metadata.draft: it is a claim the client makes (maintainer's
decision, 2026-10-18).

A model and its drafter are one unit (docs/model.md, "The drafter"):
load_model refuses a folder whose drafter is missing or was trained
against other weights, and the draft is written by speculative decoding
with that drafter, as wpt-chat writes, the scheduler's pass times
measured once when the folder is loaded.

PyTorch and the model packages are imported only when a draft is asked
for: the rest of wpt-web runs without them. One model folder stays loaded
at a time, and is loaded again when a file of the folder or of its
drafter/ changes (its weights included); one draft is written at a time. A
folder that cannot be loaded (a broken checkpoint, a drafter of other
weights, a tokenizer that does not match) is answered with 422. Only a fault
of the folder's files is: missing or malformed files, a tokenizer or drafter
that does not match, a state dict whose names or shapes do not fit. A fault
of the machine (out of memory, a GPU driver error) is not the folder's: it
is answered with 500, saying so. Any other exception (a bug, for example)
is answered with 500 too, saying the fault is unknown, with its type and
text: it blames neither the machine nor the folder (maintainer's decision,
2026-10-18).
"""

import threading
from pathlib import Path

from wpt_dataset.conversations import content_sha256

MAX_NEW_TOKENS = 200  # as wpt-chat


class Drafter:
    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = None  # (folder, state of its files, model, drafter, costs, tokenizer, weights)

    def draft(self, folder, conversation, max_new_tokens=None) -> tuple[int, dict]:
        """(HTTP status, {"message": the drafted assistant message with its
        metadata.draft}) or (status, {"problems": [...]})."""
        try:
            import torch  # noqa: F401  (the model packages need it)
            from wpt_inference import sample_tokens
            from wpt_tokenizer import Tokenizer, chat
        except ImportError as error:
            return 503, {"problems": [f"{error}: install PyTorch first (uv run scripts/setup.py)"]}
        where = Path(folder).expanduser()
        if not (where / "config.json").is_file():
            return 404, {"problems": [f"{folder} is not a WPT model folder (it has no config.json)"]}
        with self._lock:
            try:
                model, drafter, costs, tokenizer, weights = self._load(where.resolve())
            except Unloadable as error:
                cause = error.__cause__
                return 422, {"problems": [f"{type(cause).__name__}: {cause}"]}
            except Exception as error:  # not the folder's files: the machine, or an unknown fault
                if _machines_fault(error):
                    return 500, {"problems": [f"the machine failed while loading the model, not the model folder "
                                              f"({type(error).__name__}: {error})"]}
                return 500, {"problems": [f"unknown fault while loading the model; neither the machine nor the "
                                          f"model folder is known to be the cause ({type(error).__name__}: {error})"]}
            try:
                ids = chat.prompt(tokenizer, conversation)
            except (KeyError, TypeError, ValueError) as error:
                return 400, {"problems": [f"the conversation cannot be written as tokens: {error!r}"]}
            generated = []
            for next_id in sample_tokens(model, ids, max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
                                         drafter=drafter, costs=costs):
                if next_id in (chat.IM_END, Tokenizer.END):
                    break
                generated.append(next_id)
            content = chat.reply(tokenizer, generated)
        if not content:
            return 422, {"problems": ["the model wrote nothing before ending its turn; ask again"]}
        draft = {"model": where.resolve().name, "weights": weights, "sha256": content_sha256(content), "edited": False}
        return 200, {"message": {"role": "assistant", "content": content, "metadata": {"draft": draft}}}

    def _load(self, folder: Path):
        """(model, drafter, costs, tokenizer, weights) of the folder, from
        the one loaded while none of its files changed; Unloadable, caused by
        the error, when the folder cannot be loaded."""
        from wpt_inference import measure_costs
        from wpt_model import load_drafter, load_model, load_tokenizer, weight_hashes
        from wpt_tokenizer import Tokenizer
        from wpt_utils.device import select_device

        state = _files_state(folder)
        if self._loaded is None or self._loaded[:2] != (folder, state):
            self._loaded = None  # the old model goes before the new one loads
            device = select_device()
            try:
                tokenizer = Tokenizer.from_bytes(load_tokenizer(folder))
                model = load_model(folder, device)  # DrafterMismatch: its drafter is missing or of other weights
                drafter = load_drafter(folder, device)
                weights = weight_hashes(folder)
            except Exception as error:
                if not _folders_fault(error):
                    raise  # not the folder's: the machine's, or unknown
                raise Unloadable() from error
            model.eval()
            costs = measure_costs(model, 1, drafter.block)  # this machine's pass times, once per folder
            self._loaded = (folder, state, model, drafter, costs, tokenizer, weights)
        return self._loaded[2:]


def _folders_fault(error: Exception) -> bool:
    """Whether a failure to load comes from the folder's files (missing or
    malformed files, a tokenizer or drafter that does not match, a state
    dict whose names or shapes do not fit) rather than from the machine. A
    RuntimeError is the folder's only when it is torch's state dict
    mismatch: out of memory and driver errors are RuntimeErrors too."""
    from safetensors import SafetensorError

    if isinstance(error, RuntimeError):
        return str(error).startswith("Error(s) in loading state_dict")
    return isinstance(error, (OSError, ValueError, KeyError, TypeError, SafetensorError))


def _machines_fault(error: Exception) -> bool:
    """Whether a failure to load is known to come from the machine: out of
    memory, or a CUDA or HIP runtime error. Any other exception is unknown."""
    import torch

    if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    return isinstance(error, RuntimeError) and ("CUDA" in str(error) or "HIP" in str(error))


class Unloadable(Exception):
    """A model folder that cannot be loaded; its cause says why."""


def _files_state(folder: Path) -> tuple:
    """(name, size, change time) of every file of the folder and of its
    drafter/: a new or rewritten weights file, config or tokenizer changes it."""
    found = []
    for where in (folder, folder / "drafter"):
        if where.is_dir():
            for path in sorted(where.iterdir()):
                if path.is_file():
                    status = path.stat()
                    found.append((path.relative_to(folder).as_posix(), status.st_size, status.st_mtime_ns))
    return tuple(found)
