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
from the folder's own records (config.json, provenance.json, evals/*.json,
drafter/config.json and drafter/provenance.json), so they cannot disagree
with what the folder holds. The drafter's weights are the model's, under
the same license and header; its prompts are the beginnings of the
dataset's training documents, so it reads no source the model did not.
WEIGHT_METADATA goes into the header of every weight
file, so the licenses travel with the weights even alone.

OUTPUT-NOTICE and LICENSE in this package are copies of the repository's
root files; a test checks they stay identical.
"""

import json
import shutil
from importlib.resources import files
from pathlib import Path

from wpt_utils.programs import code_name

LICENSE = "AGPL-3.0-or-later"
OUTPUT_LICENSE = "CC0-1.0"
REPOSITORY = "https://github.com/Workers-Intelligence-Commons/wpt-llm"
NOTICE_NAME, CARD_NAME, LICENSE_NAME, EVALS_NAME, CREDITS_NAME = "NOTICE", "README.md", "LICENSE", "evals", "CREDITS"
DRAFTER_NAME = "drafter"  # the drafter's folder (drafter.py)
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
    drafter = folder / DRAFTER_NAME
    drafter = ({"config": _read(drafter / "config.json"), "provenance": _read(drafter / "provenance.json")}
               if (drafter / "config.json").exists() else None)
    return {LICENSE_NAME: license_text(), NOTICE_NAME: notice(provenance),
            CARD_NAME: card(config, provenance, evals, drafter)}


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
        "covers and the credit it asks for, as recorded when the dataset was built.",
        "Their license texts follow the list, each text once.",
    ]
    texts = {}  # text -> the sources and licenses that give it
    for sid, source in sorted(learned_from(provenance).items()):
        addresses = [f"{part['address']} (into {part['folder']}/)" for part in source.get("addresses", [])]
        lines += ["", sid, *(f"  address:  {address}" for address in addresses or [source["address"]]),
                  f"  revision: {source['revision']}"]
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


def card(config, provenance, evals, drafter=None) -> str:
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
        *(f"- {line}" for line in _built_from(provenance)),
        f"- Device: {provenance['device']}",
        f"- Every input and tool version: [provenance.json](provenance.json)", "",
        "The steps, in order, each on the weights the one before it left:", "",
        *(f"{number}. {_step(step)}" for number, step in enumerate(provenance["steps"], start=1)), "",
        *_drafter(drafter),
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
                  f"`{result['file']}` (sha256 `{result['file_sha256'][:12]}`), {_eval_code(result)}, "
                  f"on {result['device']}. Data: [evals/{result['name']}.json](evals/{result['name']}.json)", ""]
    return "\n".join(lines)


def _eval_code(result) -> str:
    """The code a result was measured with: code() of wpt_utils.programs
    in a result that names its code, wpt-eval's package version in an
    older one that names it."""
    return f"code {code_name(result['code'])}" if "code" in result else f"wpt-eval {result['wpt_eval']}"


def learned_from(provenance) -> dict:
    """Every source the model learned from: those of every step. The
    drafter adds none: its prompts are the beginnings of the dataset's
    training documents."""
    return provenance["sources"]


def _drafter(drafter):
    if drafter is None:
        return []
    config, provenance = drafter["config"], drafter["provenance"]
    lines = ["## Drafter", "",
             f"A DSpark drafter for speculative decoding (`drafter/`): {config['layers']} layers that read the "
             f"hidden states of layers {', '.join(map(str, config['features']))}, up to {config['block']} drafts per "
             "round. The model checks every draft and keeps it only by the rule that leaves its own distribution "
             "unchanged: the drafter changes how fast the model writes, never what it writes.", ""]
    trained, target = provenance.get("trained_by"), (provenance.get("target") or {}).get("weights")
    if trained:
        dataset = provenance.get("dataset")
        lines.append(f"- Trained by `{trained['command']}`, recipe `{trained['recipe_name']}`, the model frozen, on "
                     "answers the model wrote itself to the beginnings of the training documents of dataset "
                     f"`{(dataset or 'unknown')[:12]}`")
    if target:
        lines.append("- Against the weights " + ", ".join(f"`{name}` (sha256 `{sha[:12]}`)"
                                                           for name, sha in target.items()))
    measure = provenance.get("acceptance")
    if measure:
        lines.append(f"- On held-out prompts: {measure['accepted']} of {measure['verified']} drafts kept, "
                     f"{measure['tokens_per_round']:.2f} tokens per pass of the model")
    return lines + [""]


TEXT = "windows"  # what a step that reads text reads, as provenance.json's steps say (`reads`)


def _model_kind(provenance):
    if all(step["reads"] == TEXT for step in provenance["steps"]):
        return ["A WPT mixture-of-experts model (KDA, CSA and HCA attention, mHC residual",
                "streams, Engram memory). It is trained on text only: it continues text and",
                "follows no instructions.", ""]
    return ["A WPT mixture-of-experts model (KDA, CSA and HCA attention, mHC residual",
            "streams, Engram memory), trained by the steps below.", ""]


def _step(step):
    """One line of the card for a step of provenance.json's `steps`."""
    stages = step["stages"]
    limit = sum(stage["steps"] for stage in stages)
    text = (f"`{step['step']['name']}`: method {step['method']['name']}, optimizer {step['optimizer']['name']}, "
            f"environment {step['environment']['name']}"
            + (f", locked {', '.join(step['locked'])}" if step["locked"] else "")
            + f"; {step['updates']:,} updates (limit {limit:,}), the weights of update {step['weights_update']:,}")
    if any(stage["block_size"] is not None for stage in stages):
        text += "; " + ", ".join(f"{stage['block_size']:,}-token windows, {stage['batch_size']:,} an update, for "
                                 f"{stage['steps']:,} updates" for stage in stages)
    else:
        text += f"; {stages[0]['batch_size']:,} examples an update"
        if step["sources"]:
            text += f"; trained on {', '.join(step['sources'])}"
    if step["best_validation_loss"] is not None:
        text += f"; best validation loss {step['best_validation_loss']:.4f}"
    if step["reused"]:
        text += f"; the result of an earlier run (`{step['result'][:12]}`)"
    return text + "."


def _built_from(provenance):
    commit = provenance.get("repository_commit")
    if commit is None:
        return ["Built outside a git checkout: the repository commit is unknown."]
    dirty = " with uncommitted changes" if provenance.get("repository_dirty") else ""
    return [f"Built from repository commit {commit}{dirty}."]


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))
