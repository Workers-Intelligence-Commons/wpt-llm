"""Move data files to their current shape, so no file is edited by hand.

No file carries a shape number. A step recognizes the old shape by its
fields: it says whether a document has the old shape (`applies`) and
rewrites it (`rewrite`). A document that matches the current schema needs
no step. To change a shape: update the schema file and register a step
here, in the list of its kind:

    STEPS["source"].append(Step("licenses without a credit",
                                lambda doc: any("attribution" not in entry for entry in doc["licenses"]),
                                lambda doc, where: {...}))

The change must leave a field difference for `applies` to find. A step gets
where the file lives too (repo.Located: its name for messages, and the text
of a file next to it, for the rare step that reads one).

`wpt-dataset migrate` then rewrites every file a step applies to, and
checks the result against the schema. A file migrate cannot rewrite, such as
a source's chat file (upstream's bytes), is brought to the current shape in
memory by upgrade() each time it is read.
"""

from typing import Callable, NamedTuple

from .errors import DatasetError, OutdatedSchema
from .repo import Repo

KINDS_WITH_STEPS = ("source", "mixture", "model", "tokenizer", "lock", "conversation", "task", "rating",
                    "image-lock", "image-licenses", "image-tools", "reviewed-license-texts")


class Step(NamedTuple):
    """One rewrite. what: the old shape, as a sentence part ("licenses
    without a credit"). applies(document): whether the document has it.
    rewrite(document, where): the document in the shape after it. after(where,
    document): what a rewritten file leaves to do beside it (optional)."""
    what: str
    applies: Callable
    rewrite: Callable
    after: Callable | None = None


STEPS: dict[str, list[Step]] = {kind: [] for kind in KINDS_WITH_STEPS}


def _entries(document, key) -> list[dict]:
    """The dict items of the list a document holds under key ([] when it holds none)."""
    found = document.get(key) if isinstance(document, dict) else None
    return [entry for entry in found if isinstance(entry, dict)] if isinstance(found, list) else []


def _source_attribution(document, _where):
    """A license entry gets the credit it asks for. Only a person can write
    it, so the step leaves it empty and the check that follows names each
    file to fill in."""
    return {**document, "licenses": [{**entry, "attribution": entry.get("attribution", "")}
                                     for entry in document["licenses"]]}


def _source_formats(document, _where):
    """How each file becomes Markdown for training (`formats`). Only a person
    knows each file's format, so the step leaves the list empty and the
    check that follows names each file to fill in."""
    return {**document, "formats": []}


def _license_file(entry, where) -> tuple[str, str] | None:
    """(name, text) of the LicenseRef-<name>.txt next to source.json that an
    earlier shape kept for a LicenseRef license, or None."""
    for alternative in entry["license"].split(" OR "):
        name = f"{alternative.partition(' WITH ')[0]}.txt"
        text = where.sibling_text(name) if alternative.startswith("LicenseRef-") else None
        if text is not None:
            return name, text
    return None


def _source_text(document, where):
    """Every license's text is in its entry (`text`), so that it travels into
    each dataset record and model NOTICE. A LicenseRef text moves in from its
    file; any other text is only on the source's side, so the step leaves it
    empty and the check that follows names each file to fill in."""
    licenses = []
    for entry in document["licenses"]:
        found = _license_file(entry, where)
        licenses.append({**entry, "text": entry["text"] if "text" in entry else found[1] if found else ""})
    return {**document, "licenses": licenses}


def _drop_moved_license_files(where, document):
    """Remove the LicenseRef files whose text the written source.json now
    holds exactly; anything else stays for the check to report."""
    for entry in document["licenses"]:
        found = _license_file(entry, where)
        if found and found[1] == entry["text"]:
            where.remove_sibling(found[0])


def _old_decision(document) -> bool:
    decision = document.get("decision")
    return isinstance(decision, dict) and (decision.get("decision") == "retired" or (
        decision.get("decision") == "refused" and decision.get("reason") == "under review"))


def _source_decision(document, _where):
    """A source has exactly three states, "under-review", "admitted" and
    "refused": a refusal whose reason was "under review" becomes
    "under-review" without that reason, and "retired" becomes "refused",
    keeping its reason (which names the source that replaced it)."""
    decision = document["decision"]
    if decision["decision"] == "refused" and decision.get("reason") == "under review":
        decision = {key: value for key, value in decision.items() if key != "reason"} | {"decision": "under-review"}
    elif decision["decision"] == "retired":
        decision = {**decision, "decision": "refused"}
    return {**document, "decision": decision}


# the tags that named a training step, and the tags that name the step doing the same
TAGS_OLD = {"sft": "answer-as-assistant", "grpo": "finish-tasks", "token-efficiency": "finish-tasks-with-less-writing"}


def _old_tags(document) -> bool:
    return any(isinstance(tag, str) and tag in TAGS_OLD for entry in _entries(document, "tags")
               for tag in (entry.get("tags") if isinstance(entry.get("tags"), list) else ()))


def _source_tags(document, _where):
    """A tag that picks a training step's examples is the step's name: sft
    becomes answer-as-assistant, grpo finish-tasks and token-efficiency
    finish-tasks-with-less-writing (TAGS_OLD), in every tag list; a tag
    listed twice by the rename stays once."""
    return {**document, "tags": [{**entry, "tags": list(dict.fromkeys(TAGS_OLD.get(tag, tag) for tag in entry["tags"]))}
                                 for entry in document["tags"]]}


STEPS["source"] += [
    Step("a license entry without its credit", lambda d: any("attribution" not in e for e in _entries(d, "licenses")),
         _source_attribution),
    Step("no formats", lambda d: "formats" not in d, _source_formats),
    Step("a license entry without its text", lambda d: any("text" not in e for e in _entries(d, "licenses")),
         _source_text, _drop_moved_license_files),
    Step("a decision of an earlier kind", _old_decision, _source_decision),
    Step("a tag that names a step by its old name", _old_tags, _source_tags),
]

# the numbers of the architecture an earlier shape did not hold
NEW_ATTENTION = ("attention_heads", "attention_head_dim", "query_rank", "output_groups", "output_rank", "index_heads",
                 "index_head_dim", "index_top_k")


def _architecture(document, key) -> bool:
    return isinstance(document.get("model"), dict) and key in document["model"]


def _model_heads(document, _where):
    """The architecture's numbers are those of docs/model.md. The window, the
    layers and the width carry over, and the old heads become both attention
    kinds' heads; the other numbers are a person's choice, so the step
    leaves them at 0 and the check that follows names each file to fill in."""
    old = document["model"]
    head_dim = old["n_embd"] // old["n_head"]
    model = {"block_size": old["block_size"], "n_layer": old["n_layer"], "d_model": old["n_embd"],
             "kda_heads": old["n_head"], "kda_head_dim": head_dim, "mla_heads": old["n_head"], "mla_head_dim": head_dim,
             "mla_kv_rank": 0, "n_experts": 0, "top_k": 0, "expert_width": 0, "latent_width": 0, "shared_width": 0,
             "engram_layers": [], "engram_order": 0, "engram_heads": 0, "engram_dim": 0, "engram_table_size": 0}
    return {**document, "model": model}


def _model_attention(document, _where):
    """The attention is DeepSeek-V4's CSA and HCA (docs/model.md,
    "Attention"), whose numbers are new: the MLA numbers go, and the new ones
    are a person's choice, as is which global layer is CSA and which HCA
    (global_attention), which gated MLA's numbers do not say. So the step
    leaves the numbers at 0 and the list empty, and the check that follows
    names each file to fill in. Weights trained with the old attention do
    not load into the new model."""
    model = {}
    for key, value in document["model"].items():
        if key == "mla_heads":
            model.update(dict.fromkeys(NEW_ATTENTION, 0))
        elif key not in ("mla_head_dim", "mla_kv_rank"):
            model[key] = value
        if key == "n_layer":
            model["global_attention"] = []
    return {**document, "model": model}


def _model_stages(document, _where):
    """The model trains in stages (docs/training.md, "Stages"), each with its
    window length, steps and batch: the model's block_size and the
    training's steps and batch_size become one stage, in the training part
    where steps were."""
    model = {key: value for key, value in document["model"].items() if key != "block_size"}
    old, training = document["training"], {}
    for key, value in old.items():
        if key == "steps":
            training["stages"] = [{"block_size": document["model"]["block_size"], "steps": value,
                                   "batch_size": old["batch_size"]}]
        elif key != "batch_size":
            training[key] = value
    return {**document, "model": model, "training": training}


# the drafter's training values every recipe states
DRAFTER_DEFAULTS = {"prompts": 256, "prompt_tokens": 64, "answer_tokens": 256, "epochs": 10, "learning_rate": 6e-4,
                    "held_out": 0.1, "anchors": 512, "batch": 8, "seed": 0}


def _model_drafter(document, _where):
    """The `drafter` section (docs/model.md, "The drafter"): the drafter that
    wpt-train trains at the end of every run. It goes in after `training`,
    with the training values every recipe states (DRAFTER_DEFAULTS); the
    settings it leaves out keep their defaults."""
    migrated = {}
    for key, value in document.items():
        migrated[key] = value
        if key == "training":
            migrated["drafter"] = dict(DRAFTER_DEFAULTS)
    return migrated


# the settings of the old `muon` that make the loss, which the read-text step's method holds
NEXT_TOKEN_OLD = ("mtp_weight", "mtp_late_weight", "mtp_switch")
# the old post-training steps and the training steps that do the same, and the settings of theirs that
# belong to the step's optimizer (adamw), under that optimizer's names
STEPS_OLD = {"sft": "answer-as-assistant", "grpo": "finish-tasks", "token-efficiency": "finish-tasks-with-less-writing"}
OPTIMIZER_OLD = {"learning_rate": "lr", "weight_decay": "weight_decay"}


def _model_steps(document, _where):
    """A model trains in one ordered list of steps (docs/training.md,
    "Steps"), in place of `training` and `posttrain`: the training becomes
    the first step, read-text, its method the step's optimizer (with its
    settings but those of the loss, NEXT_TOKEN_OLD, which become the step's),
    its stages, eval_every and eval_batches the step's settings, nothing
    locked; its seed becomes the recipe's `seed`. Each post-training step
    follows under its new name (STEPS_OLD), the settings of its learning
    rate and weight decay given to its optimizer, adamw, and a `tag` setting
    that names an old step renamed with it; a step given as a Python file
    keeps its name and settings, and the check that follows says what the
    file lacks. `steps` and `seed` go where `training` was, `posttrain`
    goes."""
    training = document["training"]
    method_settings = training["settings"]
    loss = {key: value for key, value in method_settings.items() if key in NEXT_TOKEN_OLD}
    first = {"step": "read-text",
             "optimizer": {"name": training["method"],
                           "settings": {key: value for key, value in method_settings.items() if key not in loss}},
             "settings": {"stages": training["stages"], "eval_every": training["eval_every"],
                          "eval_batches": training["eval_batches"], **loss},
             "locked": []}
    steps = [first]
    for entry in document["posttrain"]:
        name, settings = (entry, {}) if isinstance(entry, str) else (entry["step"], entry["settings"])
        if name not in STEPS_OLD:
            steps.append({"step": name, "settings": settings})
            continue
        optimizer = {OPTIMIZER_OLD[key]: value for key, value in settings.items() if key in OPTIMIZER_OLD}
        # a tag that names an old step names the step that does the same
        step = {"step": STEPS_OLD[name], "settings": {key: STEPS_OLD.get(value, value) if key == "tag" else value
                                                      for key, value in settings.items() if key not in OPTIMIZER_OLD}}
        steps.append({**step, **({"optimizer": {"name": "adamw", "settings": optimizer}} if optimizer else {})})
    migrated = {}
    for key, value in document.items():
        if key == "training":
            migrated["seed"], migrated["steps"] = training["seed"], steps
        elif key != "posttrain":
            migrated[key] = value
    return migrated


STEPS["model"] += [
    Step("an architecture with n_embd", lambda d: _architecture(d, "n_embd"), _model_heads),
    Step("an architecture with gated MLA", lambda d: _architecture(d, "mla_heads"), _model_attention),
    Step("a window length in the architecture", lambda d: _architecture(d, "block_size") and "training" in d,
         _model_stages),
    Step("no drafter section", lambda d: "training" in d and "drafter" not in d, _model_drafter),
    Step("training and post-training apart", lambda d: "training" in d and "posttrain" in d, _model_steps),
]


STEPS["task"] += [
    Step("replies without the paths their answers live in", lambda d: bool(d.get("replies")) and "answered_in" not in d,
         lambda d, _where: {**d, "answered_in": []}),
]
# `answered_in` is the paths where a task's answers live, required with replies. Only a person knows where they
# live, so the step leaves it empty, and the check that follows names each file to fill in.


STEPS["tokenizer"] += [
    Step("no vocabulary version", lambda d: "version" not in d, lambda d, _where: {**d, "version": 1}),
]


def _lock_without_documents(document, _where):
    """A lock that does not list the documents its vocabulary was trained on
    cannot be converted."""
    raise OutdatedSchema("the lock does not list the documents its vocabulary was trained on, "
                         "so it cannot be converted: delete it and build again")


STEPS["lock"] += [
    Step("no list of the documents its vocabulary was trained on", lambda d: "documents" not in d,
         _lock_without_documents),
    Step("the tokenizer code's version", lambda d: "wpt_tokenizer" in d,
         lambda d, _where: {key: value for key, value in d.items() if key != "wpt_tokenizer"}),
]

# an earlier shape carried a number in `schema`; no file does now
for _kind in KINDS_WITH_STEPS:
    STEPS[_kind].append(Step("a shape number (`schema`)", lambda d: "schema" in d,
                             lambda d, _where: {key: value for key, value in d.items() if key != "schema"}))


def _applied(document, kind: str, where, steps) -> tuple[object, list[Step]]:
    """(the document after every step that applies, those steps, in order);
    the document itself, and none, when it is not an object (the check then
    says what is wrong). where names it in errors and goes to each step."""
    applied = []
    for step in steps.get(kind, ()) if isinstance(document, dict) else ():
        if step.applies(document):
            try:
                document = step.rewrite(document, where)
            except DatasetError as error:
                raise type(error)(f"{where}: {error}") from None
            applied.append(step)
    return document, applied


def outdated(document, kind: str, steps=None) -> list[str]:
    """What is of an earlier shape in the document, as the steps of its kind
    name it ([]: nothing; also for what is not an object, which the check
    then reports)."""
    steps = STEPS if steps is None else steps
    if not isinstance(document, dict):
        return []
    return [step.what for step in steps.get(kind, ()) if step.applies(document)]


def upgrade(document, kind: str, path, steps=None):
    """The document brought to the current shape of its kind, step by step;
    the document itself when it is current, or not an object. path names it
    in errors and goes to each step."""
    return _applied(document, kind, path, STEPS if steps is None else steps)[0]


def migrate(repo: Repo, steps=None, log=print) -> list[str]:
    """Rewrite every file that is of an earlier shape, through the service
    layer (Repo.rewrite). Returns the places of the files changed (data/...)."""
    steps = STEPS if steps is None else steps
    changed = []
    for kind, name in repo.versioned():
        where = repo.located(kind, name)
        migrated, applied = _applied(repo.raw(kind, name), kind, where, steps)
        if not applied:
            continue
        changed.append(repo.rewrite(kind, name, migrated))
        for step in applied:
            if step.after is not None:
                step.after(where, migrated)
        log(f"{changed[-1]}: {'; '.join(step.what for step in applied)}")
    if steps is STEPS:  # the real shapes: every file must now validate
        for kind, name in repo.versioned():
            repo.validate_file(kind, name)
    return changed
