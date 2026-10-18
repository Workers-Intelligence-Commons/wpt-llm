"""The license documents of a model folder.

Next to its weights, a model folder carries:

    LICENSE       the GNU AGPL, version 3: the license of the weights
    CREDITS/      for each source whose files hold many documents with
                  their own licenses, <source id>.jsonl: one line per
                  document, with its license and the credit its record gives
    NOTICE        the licenses of the weights and of generated text, the
                  sources the model learned from with the credit each one
                  asks for, and the output notice
    README.md     the model card: the same licenses first, then what the
                  model is, how it was built, and its evaluation results

write_documents(folder) generates them (documents(folder) gives their text)
from the folder's own records (config.json, provenance.json, evals/*.json),
so they cannot disagree with what the folder holds. WEIGHT_METADATA goes into the header of every weight
file, so the licenses travel with the weights even alone.

OUTPUT-NOTICE and LICENSE in this package are copies of the repository's
root files; a test checks they stay identical.
"""

import json
import shutil
from importlib.resources import files
from pathlib import Path

LICENSE = "AGPL-3.0-or-later"
OUTPUT_LICENSE = "CC0-1.0"
REPOSITORY = "https://github.com/Workers-Intelligence-Commons/wpt-llm"
NOTICE_NAME, CARD_NAME, LICENSE_NAME, EVALS_NAME, CREDITS_NAME = "NOTICE", "README.md", "LICENSE", "evals", "CREDITS"
WEIGHT_METADATA = {"license": LICENSE, "output_license": OUTPUT_LICENSE,
                   "notice": f"NOTICE in the model folder; {REPOSITORY}"}
SUMMARY = (f"Weights: {LICENSE}. Generated text: {OUTPUT_LICENSE}, waived by the authors of WPT; "
           "material it reproduces from its training data or its context (prompt, documents, tool "
           "results) keeps its own license.")


def output_notice() -> str:
    return (files(__package__) / "OUTPUT-NOTICE").read_text(encoding="utf-8")


def license_text() -> str:
    return (files(__package__) / "LICENSE").read_text(encoding="utf-8")


def documents(folder) -> dict[str, str]:
    """{LICENSE, NOTICE and README.md: their text} for a model folder, from
    its config.json, provenance.json and evals/*.json."""
    folder = Path(folder)
    config = _read(folder / "config.json")
    provenance = _read(folder / "provenance.json")
    evals = [_read(path) for path in sorted((folder / EVALS_NAME).glob("*.json"))]
    return {LICENSE_NAME: license_text(), NOTICE_NAME: notice(provenance), CARD_NAME: card(config, provenance, evals)}


def write_documents(folder):
    """Write documents(folder) into the folder. Replaces earlier copies."""
    for name, text in documents(folder).items():
        (Path(folder) / name).write_text(text, encoding="utf-8", newline="\n")


def write_credits(folder, provenance, read_object):
    """Write CREDITS/<source id>.jsonl for every source the model learned
    from whose dataset record lists `credits`; read_object(sha256) gives
    that object's bytes. Replaces earlier copies."""
    folder = Path(folder) / CREDITS_NAME
    shutil.rmtree(folder, ignore_errors=True)
    for sid, source in sorted(learned_from(provenance).items()):
        if source.get("credits"):
            folder.mkdir(exist_ok=True)
            (folder / f"{sid}.jsonl").write_bytes(read_object(source["credits"]))


def notice(provenance) -> str:
    lines = [
        "WPT model notice", "",
        f"Weights: {LICENSE}. These weights are part of WPT (Workers' Pre-trained",
        "Transformer) and are licensed under the GNU Affero General Public License,",
        "version 3 or any later version; the license text is in LICENSE, next to",
        "this file.", "",
        f"Generated text: {OUTPUT_LICENSE}, under the output notice at the end of this file.", "",
        f"Project: {REPOSITORY}",
        *_built_from(provenance), "", "",
        "Sources this model learned from", "",
        "Each source with its exact revision, and for each license the paths it",
        "covers and the credit it asks for, as recorded when the dataset was built",
        "or when a post-training step read it. Their license texts follow the list,",
        "each text once.",
    ]
    texts = {}  # text -> the sources and licenses that give it
    for sid, source in sorted(learned_from(provenance).items()):
        lines += ["", sid, f"  address:  {source['address']}", f"  revision: {source['revision']}"]
        if source.get("credits"):
            lines.append(f"  credit of each document: {CREDITS_NAME}/{sid}.jsonl")
        for entry in source["licenses"]:
            covers = ", ".join(entry["paths"]) + (f" (records whose license is {', '.join(map(repr, entry['values']))})"
                                                  if entry.get("values") else "")
            lines += [f"  {entry['license']} for {covers}:",
                      *(f"    {line}" for line in entry["attribution"].splitlines())]
            if entry.get("text"):  # datasets built before shape 4 recorded no texts
                texts.setdefault(entry["text"], []).append(f"{sid} ({entry['license']})")
    if texts:
        lines += ["", "", "License texts"]
        for text, users in texts.items():
            lines += ["", f"--- {', '.join(users)}", "", text.rstrip("\n")]
    return "\n".join(lines) + "\n\n\n" + output_notice()


def card(config, provenance, evals) -> str:
    lines = [
        "---", "license: agpl-3.0", "---", "",
        f"# WPT model, dataset {provenance['dataset'][:12]}", "",
        f"> {SUMMARY}",
        ">",
        "> The full notice, with every source and the credit it asks for, is in",
        f"> [NOTICE](NOTICE). Project: {REPOSITORY}", "",
        "## Model", "",
        *_model_kind(provenance),
        f"- Parameters: {provenance['parameters']:,}",
        f"- Layers: {config['n_layer']}, width {config['d_model']}; experts: {config['n_experts']} routed "
        f"(top {config['top_k']}) and 2 shared per layer; Engram memory in layers "
        f"{', '.join(map(str, config['engram_layers'])) or 'none'}",
        f"- Context: {config['block_size']} tokens; vocabulary: {config['vocab_size']:,} tokens", "",
        "## Training", "",
        f"- Dataset: `{provenance['dataset']}`",
        f"- {_steps(provenance)}, {_batch(provenance)}",
        f"- Method: {provenance['method']['name']} "
        f"({', '.join(f'{k} {v}' for k, v in sorted(provenance['method']['settings'].items()))})",
        f"- Best validation loss: {provenance['best_validation_loss']:.4f}",
        *(f"- {line}" for line in _built_from(provenance)),
        f"- Device: {provenance['device']}",
        f"- Every input and tool version: [provenance.json](provenance.json)", "",
        *_post_training(provenance.get("post_training")),
        "## Sources", "",
        "| Source | Revision | Licenses |", "|---|---|---|",
        *(f"| {sid} | `{source['revision']}` | {', '.join(sorted({e['license'] for e in source['licenses']}))} |"
          for sid, source in sorted(learned_from(provenance).items())), "",
        "Authors and the credit each source asks for: [NOTICE](NOTICE).", "",
        "## Evaluations", "",
    ]
    if not evals:
        lines += ["None yet: run `wpt-eval <model folder>`.", ""]
    for result in evals:
        lines += [f"### {result['name']}", "", result["text"].rstrip(), "",
                  f"`{result['file']}` (sha256 `{result['file_sha256'][:12]}`), wpt-eval {result['wpt_eval']}, "
                  f"on {result['device']}. Data: [evals/{result['name']}.json](evals/{result['name']}.json)", ""]
    return "\n".join(lines)


def learned_from(provenance) -> dict:
    """Every source the model learned from: the pre-training dataset's, then
    those post-training steps read."""
    return {**provenance["sources"], **(provenance.get("post_training") or {}).get("sources", {})}


def _post_training(post_training):
    if not post_training:
        return []
    return ["## Post-training", "",
            "Trained from the base weights "
            + ", ".join(f"`{name}` (sha256 `{sha[:12]}`)" for name, sha in post_training["base_weights"].items())
            + ", then changed by these steps, in order; every one succeeded:", "",
            *(f"{i}. `{step['file']}` (sha256 `{step['file_sha256'][:12]}`)"
              + (f", trained on {', '.join(step['sources'])}" if step.get("sources") else "")
              for i, step in enumerate(post_training["steps"], start=1)),
            "", *(f"- {line}" for line in _built_from(post_training)), ""]


def _model_kind(provenance):
    if provenance.get("post_training"):
        return ["A WPT mixture-of-experts model (KDA, CSA and HCA attention, mHC residual",
                "streams, Engram memory), pre-trained and then changed by the post-training",
                "steps below.", ""]
    return ["A WPT mixture-of-experts model (KDA, CSA and HCA attention, mHC residual",
            "streams, Engram memory). It is pre-trained only: it continues text and follows",
            "no instructions.", ""]


def _steps(provenance):
    training = provenance["recipe"]["training"]
    stages = training.get("stages")  # recipes before shape 4 had one window length and its steps
    limit = sum(stage["steps"] for stage in stages) if stages else training["steps"]
    if "steps_trained" not in provenance:  # trained before the steps were recorded
        return f"Steps: at most {limit:,}"
    text = (f"Steps: {provenance['steps_trained']:,} trained (limit {limit:,}); "
            f"the weights are those of step {provenance['weights_step']:,}")
    if stages and len(stages) > 1:
        text += "; stages: " + ", ".join(f"{stage['block_size']:,}-token windows"
                                         + (f", {stage['batch_size']:,} a step" if "batch_size" in stage else "")
                                         + f", for {stage['steps']:,} steps" for stage in stages)
    return text


def _batch(provenance):
    batch = provenance["batch"]
    if isinstance(batch, dict):  # trained before each stage had its own batch: one for the run
        return f"batch {batch['windows']:,} windows"
    sizes = [stage["windows"] for stage in batch]
    if len(set(sizes)) == 1:
        return f"batch {sizes[0]:,} windows"
    return "batches of " + ", ".join(f"{size:,}" for size in sizes) + " windows, by stage"


def _built_from(provenance):
    commit = provenance.get("repository_commit")
    if commit is None:
        return ["Built outside a git checkout: the repository commit is unknown."]
    dirty = " with uncommitted changes" if provenance.get("repository_dirty") else ""
    return [f"Built from repository commit {commit}{dirty}."]


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))
