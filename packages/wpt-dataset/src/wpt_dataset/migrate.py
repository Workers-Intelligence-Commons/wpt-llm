"""Move data files from one shape to the next, so no file is edited by hand.

Each migration turns a document of shape N into shape N+1. To change a
shape: bump CURRENT_SCHEMA in repo.py, update the schema file, and register
the step here:

    MIGRATIONS["source"][1] = lambda doc, path: {**doc, "schema": 2, ...}

A step gets where the file lives too (repo.Located: its name for
messages, and the text of a file next to it, for the rare step that reads
one).

`wpt-dataset migrate` then rewrites every file that is behind, and checks
the result against the new schema. A file migrate cannot rewrite, such as
a source's chat file (upstream's bytes), is brought to the current shape in
memory by upgrade() each time it is read.
"""

from .errors import DatasetError, OutdatedSchema
from .repo import CURRENT_SCHEMA, Repo

MIGRATIONS: dict[str, dict[int, callable]] = {kind: {} for kind in CURRENT_SCHEMA}


def _source_1_to_2(document, _path):
    """Shape 2 gives every license entry the credit it asks for. Only a
    person can write it, so the step leaves it empty and the check that
    follows names each file to fill in."""
    licenses = [{**entry, "attribution": entry.get("attribution", "")} for entry in document["licenses"]]
    return {**document, "schema": 2, "licenses": licenses}


MIGRATIONS["source"][1] = _source_1_to_2


def _source_2_to_3(document, _path):
    """Shape 3 says how each file becomes Markdown for training (`formats`).
    Only a person knows each file's format, so the step leaves the list
    empty and the check that follows names each file to fill in."""
    return {**document, "schema": 3, "formats": document.get("formats", [])}


MIGRATIONS["source"][2] = _source_2_to_3


def _license_file(entry, where) -> tuple[str, str] | None:
    """(name, text) of the LicenseRef-<name>.txt next to source.json that
    shape 3 kept for a LicenseRef license, or None."""
    for alternative in entry["license"].split(" OR "):
        name = f"{alternative.partition(' WITH ')[0]}.txt"
        text = where.sibling_text(name) if alternative.startswith("LicenseRef-") else None
        if text is not None:
            return name, text
    return None


def _source_3_to_4(document, where):
    """Shape 4 keeps every license's text in its entry (`text`), so that it
    travels into each dataset record and model NOTICE. A LicenseRef text
    moves in from its file; any other text is only on the source's side, so
    the step leaves it empty and the check that follows names each file to
    fill in."""
    licenses = []
    for entry in document["licenses"]:
        found = _license_file(entry, where)
        text = found[1] if found else entry.get("text", "")
        licenses.append({**entry, "text": text})
    return {**document, "schema": 4, "licenses": licenses}


MIGRATIONS["source"][3] = _source_3_to_4


def _source_4_to_5(document, _path):
    """Shape 5 adds the huggingface type, the records format and the `values`
    of a license entry; a shape 4 file uses none of them, so only its number
    changes."""
    return {**document, "schema": 5}


MIGRATIONS["source"][4] = _source_4_to_5


def _model_1_to_2(document, _path):
    """Shape 2 holds the numbers of the architecture in docs/model.md. The
    window, the layers and the width carry over, and the old heads become
    both attention kinds' heads; the other numbers are a person's choice, so
    the step leaves them at 0 and the check that follows names each file to
    fill in."""
    old = document["model"]
    head_dim = old["n_embd"] // old["n_head"]
    model = {"block_size": old["block_size"], "n_layer": old["n_layer"], "d_model": old["n_embd"],
             "kda_heads": old["n_head"], "kda_head_dim": head_dim, "mla_heads": old["n_head"], "mla_head_dim": head_dim,
             "mla_kv_rank": 0, "n_experts": 0, "top_k": 0, "expert_width": 0, "latent_width": 0, "shared_width": 0,
             "engram_layers": [], "engram_order": 0, "engram_heads": 0, "engram_dim": 0, "engram_table_size": 0}
    return {**document, "schema": 2, "model": model}


MIGRATIONS["model"][1] = _model_1_to_2


NEW_ATTENTION = ("attention_heads", "attention_head_dim", "query_rank", "output_groups", "output_rank", "index_heads",
                 "index_head_dim", "index_top_k")


def _model_2_to_3(document, _path):
    """Shape 3 replaces gated MLA with DeepSeek-V4's CSA and HCA, whose
    numbers are new (docs/model.md, "Attention"): the MLA numbers go, and
    the new ones are a person's choice, as is which global layer is CSA and
    which HCA (global_attention), which gated MLA's numbers do not say. So
    the step leaves the numbers at 0 and the list empty, and the check that
    follows names each file to fill in. Weights trained with shape 2 do not
    load into the new model."""
    model = {}
    for key, value in document["model"].items():
        if key == "mla_heads":
            model.update(dict.fromkeys(NEW_ATTENTION, 0))
        elif key not in ("mla_head_dim", "mla_kv_rank"):
            model[key] = value
        if key == "n_layer":
            model["global_attention"] = []
    return {**document, "schema": 3, "model": model}


MIGRATIONS["model"][2] = _model_2_to_3


def _model_3_to_4(document, _path):
    """Shape 4 trains in stages (docs/training.md, "Stages"), each with its
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
    return {**document, "schema": 4, "model": model, "training": training}


MIGRATIONS["model"][3] = _model_3_to_4


# the drafter's training values every recipe states (maintainer's decision, 2026-10-18)
DRAFTER_5 = {"prompts": 256, "prompt_tokens": 64, "answer_tokens": 256, "epochs": 10, "learning_rate": 6e-4,
             "held_out": 0.1, "anchors": 512, "batch": 8, "seed": 0}


def _model_4_to_5(document, _path):
    """Shape 5 adds the `drafter` section (docs/model.md, "The drafter";
    maintainer's decisions, 2026-10-17 and 2026-10-18): the drafter that
    wpt-train and wpt-posttrain train at the end of every run. It goes in
    after `training`, with the training values every recipe states
    (DRAFTER_5); the settings it leaves out keep their defaults."""
    migrated = {}
    for key, value in document.items():
        migrated[key] = 5 if key == "schema" else value
        if key == "training":
            migrated["drafter"] = dict(DRAFTER_5)
    return migrated


MIGRATIONS["model"][4] = _model_4_to_5


def _conversation_1_to_2(document, _path):
    """Shape 2 adds the optional `metadata` object; a shape 1 conversation
    has none, so only its number changes."""
    return {**document, "schema": 2}


MIGRATIONS["conversation"][1] = _conversation_1_to_2


def _conversation_2_to_3(document, _path):
    """Shape 3 adds a message's optional `metadata`, which records a model
    draft (docs/chat-format.md, "Model drafts"); a shape 2 conversation has
    none, so only its number changes."""
    return {**document, "schema": 3}


MIGRATIONS["conversation"][2] = _conversation_2_to_3


def _drop_moved_license_files(where, document):
    """Remove the LicenseRef files whose text the written source.json now
    holds exactly; anything else stays for the check to report."""
    for entry in document["licenses"]:
        found = _license_file(entry, where)
        if found and found[1] == entry["text"]:
            where.remove_sibling(found[0])


def _tokenizer_1_to_2(document, _path):
    """Shape 2 gives the recipe a vocabulary `version`; existing ones start at 1."""
    return {**document, "schema": 2, "version": 1}


MIGRATIONS["tokenizer"][1] = _tokenizer_1_to_2


def _lock_1_to_2(document, _path):
    """Shape 2 lists the documents the vocabulary was trained on, which a
    shape 1 lock never recorded, so it cannot be converted."""
    raise OutdatedSchema("a shape 1 lock does not list the documents its vocabulary was trained on, "
                         "so it cannot become shape 2: delete it and build again")


MIGRATIONS["lock"][1] = _lock_1_to_2


def upgrade(document, kind: str, path, migrations=None, current=None):
    """The document brought to the current shape of its kind, step by step;
    the document itself when it is current, or not a versioned object (the
    check then says what is wrong). path names it in errors and goes to
    each step."""
    migrations = MIGRATIONS if migrations is None else migrations
    current = CURRENT_SCHEMA if current is None else current
    while isinstance(document, dict) and isinstance(document.get("schema"), int) \
            and document["schema"] < current[kind]:
        step = migrations[kind].get(document["schema"])
        if step is None:
            raise KeyError(f"{path}: no migration for {kind} shape {document['schema']}")
        try:
            document = step(document, path)
        except DatasetError as error:
            raise type(error)(f"{path}: {error}") from None
    return document


def migrate(repo: Repo, migrations=None, current=None, log=print) -> list[str]:
    """Rewrite every file whose shape is behind, through the service layer
    (Repo.rewrite). Returns the places of the files changed (data/...)."""
    migrations = MIGRATIONS if migrations is None else migrations
    current = CURRENT_SCHEMA if current is None else current
    changed = []
    for kind, name in repo.versioned():
        document, where = repo.raw(kind, name), repo.located(kind, name)
        start = document.get("schema") if isinstance(document, dict) else None
        document = upgrade(document, kind, where, migrations, current)
        if isinstance(document, dict) and document.get("schema") != start:
            changed.append(repo.rewrite(kind, name, document))
            if kind == "source" and start < 4 <= document["schema"]:
                _drop_moved_license_files(where, document)
            log(f"{changed[-1]}: shape {start} -> {document['schema']}")
    if current is CURRENT_SCHEMA:  # the real shapes: every file must now validate
        for kind, name in repo.versioned():
            repo.validate_file(kind, name)
    return changed
