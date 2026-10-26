"""The text measures (calibration, reading_gain, bible_verses) on a stand-in
model that knows the right answers, so each result shows what the measure
counts; the generation, the chat template, the confidence question and the
check library's rules are the real ones."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wpt_dataset import manifest
from wpt_agent.loop import CONFIDENCE
from wpt_dataset.build import Document, is_held_out
from wpt_dataset.repo import Repo, address_hash, dump_json
from wpt_dataset import download
from wpt_dataset.sources import fetch, held_out_group
from wpt_eval import Context, find, open_context
from wpt_eval.asking import CONTINUE, Question
from wpt_store import open_store
from wpt_store.shards import write_shard
from wpt_tokenizer import END, Tokenizer, chat

ROOT = Path(__file__).resolve().parents[3]  # the repository
AGENT = ROOT / "data" / "images" / "agent"  # the image holding the check library
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}
WORDS = "amber birch cedar dune ember fjord grove heron iris juniper kestrel lichen meadow nettle orchid".split()


def paragraph(name: str, n: int) -> str:
    """A plain paragraph of about 300 bytes, its words told apart by name and n."""
    return " ".join(f"{name}{n} {WORDS[(n + i) % len(WORDS)]} {WORDS[(n * 3 + i) % len(WORDS)]}" for i in range(20)) + "."


def document(name: str, paragraphs: int) -> str:
    return "\n\n".join(paragraph(name, n) for n in range(paragraphs)) + "\n"


class Oracle(torch.nn.Module):
    """A stand-in model: after a chat prompt, it writes the assistant blocks
    that answer(the prompt's text) gives, then <|im_end|>; a large score
    on that one token, so sampling takes it."""

    def __init__(self, tokenizer, answer, block_size=1000):
        super().__init__()
        self.tokenizer, self.answer, self.replies = tokenizer, answer, {}
        self.config = SimpleNamespace(block_size=block_size)
        self.weight = torch.nn.Parameter(torch.zeros(1))  # a device for the writer

    def reply(self, prompt: tuple) -> list[int]:
        if prompt not in self.replies:
            blocks = self.answer(chat.show(self.tokenizer, list(prompt)))
            ids = chat.encode(self.tokenizer, {"messages": [{"role": "assistant", "content": blocks}]})[0]
            start = ids.index(chat.IM_START) + len(self.tokenizer.encode(b"assistant\n")) + 1
            self.replies[prompt] = ids[start:ids.index(chat.IM_END, start) + 1]
        return self.replies[prompt]

    def forward(self, ids):
        scores = torch.zeros(ids.size(0), 1, self.tokenizer.vocab_size)
        header = len(self.tokenizer.encode(b"assistant\n"))
        for row, seen in enumerate(ids.tolist()):
            start = len(seen) - 1 - seen[::-1].index(chat.IM_START)
            reply, written = self.reply(tuple(seen[:start])), len(seen) - start - 1 - header
            scores[row, 0, reply[written] if written < len(reply) else chat.IM_END] = 1e4
        return scores


def built_in(name):
    """A built-in test's module, as wpt-eval loads it."""
    _, path, _ = find(name)
    spec = importlib.util.spec_from_file_location(f"wpt_eval_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(context, name):
    """The Result of a built-in test (the command's writing of it into a
    model folder is test_eval's)."""
    return built_in(name).run(context)


def said(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def knowing(texts, confidence=0.9):
    """An answer function that gives the rest of any paragraph of texts whose
    start the prompt quotes after "Continue this passage: ", and a wrong
    answer otherwise; asked the confidence question after it, it replies
    confidence, or 0.2 for the wrong answer, from 0 to 100."""
    paragraphs = [p.strip() for text in texts for p in text.split("\n\n")]

    def answer(prompt):
        quoted = prompt.split(CONTINUE)[-1].split("<|im_end|>")[0]
        known = next((p for p in paragraphs if p.startswith(quoted)), None)
        if prompt.split("<|im_start|>user\n")[-1].startswith(CONFIDENCE):
            return said(f"{round((0.2 if known is None else confidence) * 100)}")
        return said("I do not know this passage." if known is None else known[len(quoted):].strip())
    return answer


def context_of(tmp_path, tokenizer, model, training=(), held_out=(), held_share=0.5):
    """A Context whose dataset is these documents: training as (id, text,
    passes), each repeated passes times, held_out as (id, text)."""
    tmp_path.mkdir(exist_ok=True)
    store = open_store((tmp_path / "store").as_uri())

    def shard(rows, name):
        path = tmp_path / f"{name}.parquet"
        digest = write_shard(path, [(np.asarray(tokenizer.encode(text.encode()) + [END], dtype="<u4"),
                                     doc.split("/")[0], doc, "all", n) for doc, text, n in rows])
        return [{"object": store.put_file(path), "token_sha256": digest}] if rows else []

    provenance = {"shards": shard([(d, t, n) for d, t, passes in training for n in range(passes)], "training"),
                  "held_out_shards": shard([(d, t, 0) for d, t in held_out], "held"),
                  "recipe": {"data": {"held_out": held_share}}}
    return Context(tmp_path, model, tokenizer, torch.device("cpu"), provenance, store)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The working folder: a repository whose agent image holds the
    repository's check library, and no source."""
    root = tmp_path / "wpt"
    (root / "data" / "sources").mkdir(parents=True)
    (root / "data" / "images" / "agent").mkdir(parents=True)
    for name in ("Dockerfile", "wpt_check.py"):
        (root / "data" / "images" / "agent" / name).write_bytes((AGENT / name).read_bytes())
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def tokenizer():
    return Tokenizer.train([document(name, 20).encode() for name in ("t", "h", "w")], 700)


def test_calibration_scores_each_half_and_names_how_often_training_saw_each_document(tokenizer, tmp_path, root):
    # given a model that knows its training paragraphs (confidence 0.9) and
    # answers held-out ones wrong (confidence 0.2), the training half is all
    # right with a Brier score of 0.01 and the held-out half all wrong with
    # 0.04; the training half asks only about documents the dataset holds
    # more than once (issue #28), each question naming its document's passes
    training = [(f"src/t{i}.md", document(f"t{i}x", 3), passes) for i, passes in enumerate((1, 2, 3))]
    held_out = [(f"src/h{i}.md", document(f"h{i}x", 3)) for i in range(3)]
    model = Oracle(tokenizer, knowing([text for _, text, _ in training]))
    data = measure(context_of(tmp_path, tokenizer, model, training, held_out), "calibration").data

    assert data["training"]["questions"] == 2 and data["held_out"]["questions"] == 3
    assert data["training"]["accuracy"] == 1 and data["training"]["brier"] == pytest.approx(0.01)
    assert data["held_out"]["accuracy"] == 0 and data["held_out"]["brier"] == pytest.approx(0.04)
    assert data["training"]["mean_confidence"] == pytest.approx(0.9)
    # each question keeps the answer the measure built from the two replies, as a json block
    asked = next(q for q in data["questions"] if q["part"] == "held_out")["answer"]
    assert asked == {"type": "json", "value": {"answer": "I do not know this passage.", "confidence": 0.2}}
    passes = {q["document"]: q["passes"] for q in data["questions"] if q["part"] == "training"}
    assert passes == {doc: n for doc, _, n in training if n > 1}
    assert set(data["training"]["by_passes"]) == {"2", "3"}


def test_calibration_counts_a_reply_without_a_confidence_as_no_answer(tokenizer, tmp_path, root):
    # a model that answers the confidence question with words, not a number
    # from 0 to 100 (a pre-trained one, say), scores what it scores: no
    # answer, accuracy 0, Brier score 1
    training = [("src/t0.md", document("t0x", 3), 2)]
    model = Oracle(tokenizer, lambda prompt: said("the rest, and I am fairly sure"))
    data = measure(context_of(tmp_path, tokenizer, model, training), "calibration").data
    assert data["training"] | {"by_passes": None} == {"questions": 1, "answered": 0, "accuracy": 0, "brier": 1,
                                                       "mean_confidence": None, "by_passes": None}
    assert data["held_out"]["questions"] == 0
    assert data["questions"][0]["answer"] == {"type": "json", "value": None}


def test_reading_gain_gives_the_pass_rate_with_and_without_the_text_at_each_length_and_position(tokenizer, tmp_path, root):
    # given a model that finds the rest only when the document is given,
    # every length and position shows 100% with the text, 0% without, a
    # gain of 100%; both questions ask for the confidence, as calibration does
    held_out = [(f"src/w{i}.md", document(f"w{i}x", 30)) for i in range(6)]

    def answer(prompt):
        if "In the text above, the rest of one paragraph is left out" not in prompt:
            return said("30" if prompt.split("<|im_start|>user\n")[-1].startswith(CONFIDENCE) else "I cannot tell.")
        return knowing([text for _, text in held_out])(prompt)

    context = context_of(tmp_path, tokenizer, Oracle(tokenizer, answer, block_size=1200), held_out=held_out)
    data = measure(context, "reading_gain").data
    assert data["ran"] and len(data["cells"]) == 9
    assert {(cell["share"], cell["position"]) for cell in data["cells"]} == \
           {(share, position) for share in (0.25, 0.5, 1.0) for position in ("start", "middle", "end")}
    for cell in data["cells"]:
        assert cell["questions"] > 0, cell
        assert (cell["with_text"], cell["without"], cell["gain"]) == (1, 0, 1)
        # each answer is the reply's text and the confidence the second reply gave
        assert cell["answers"][0]["without"] == {"type": "json", "value": {"answer": "I cannot tell.",
                                                                           "confidence": 0.3}}


def test_reading_gain_places_the_blanked_paragraph_where_each_position_says(tokenizer):
    module = built_in("reading_gain")
    text = document("w0x", 30).encode()
    start = text.index(paragraph("w0x", 15).encode())
    span = (start, start + len(paragraph("w0x", 15)))
    cut = text.index(b" ", start + 120)
    question = Question("src/w0.md", "src", text, span, cut)
    quote = text[start:cut].decode()
    found = module.read(tokenizer, question)
    given = {position: module.given_text(found, tokenizer, 300, position)[1] for position in module.POSITIONS}
    assert given["start"].startswith(quote) and not given["start"].endswith(module.BLANK)
    assert given["end"].endswith(quote + module.BLANK)
    assert quote + module.BLANK in given["middle"] and not given["middle"].startswith(quote) \
        and not given["middle"].endswith(module.BLANK)
    for found in given.values():  # about the length asked for, never the paragraph's rest
        assert 280 <= len(tokenizer.encode(found.encode())) <= 310
        assert text[cut:span[1]].decode().strip()[:30] not in found




def real_source(sid) -> SimpleNamespace:
    """A source of the repository, its source.json read as a file, not
    through a Repo, whose guard of the repository's data folder would
    outlive the test."""
    return SimpleNamespace(id=sid, data=json.loads((ROOT / "data" / "sources" / sid / "source.json").read_text()))


# eBible's read-aloud sources hold a translation out by book and chapter
READ_ALOUD_RULE = real_source("ebible-web-07317969").data["held_out_by"]


def git_source(tmp_path, root, store, folder_name, files, data, fetched=True):
    """An admitted git source of files, its address a folder named
    folder_name (as eBible's archives are named); in store when fetched."""
    folder = tmp_path / folder_name
    folder.mkdir()
    for path, text in files.items():
        (folder / path).write_text(text, encoding="utf-8", newline="\n")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=folder, env=GIT_ENV, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True).stdout.strip()
    data = {"type": "git", "address": str(folder), "revision": commit,
            "licenses": [{"license": "MIT", "paths": ["**"], "attribution": "T", "text": "Copyright T\n"}],
            "formats": [{"format": "text", "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-23", "reason": "test"}, **data}
    staging = tmp_path / f"elsewhere-{folder_name}"
    (staging / "download").mkdir(parents=True)
    data["checksum"] = manifest.checksum(fetch(data, data["address"], download.Staging(staging / "files"),
                                               staging / "download"))
    sid = f"ebible-{folder_name.replace('_', '-').removesuffix('.zip')}-{address_hash(str(folder))}"
    (root / "data" / "sources" / sid).mkdir()
    (root / "data" / "sources" / sid / "source.json").write_text(dump_json(data), encoding="utf-8", newline="\n")
    if fetched:  # as `wpt-dataset download` would
        with store.session(f"20261025T120000Z-test-download-{sid}", "test", "abc123", False) as session:
            [done] = download.download(Repo(root), [sid], store, session, log=lambda *_: None)
        assert done.error is None, done.error
    return sid


def translation(tmp_path, root, store, code, language, verses, twin=True, fetched=True):
    """eBible's two sources of one translation: the numbered one,
    <code>_vpl.zip, one verse per line, tagged verse-numbers only; and,
    with twin, its read-aloud twin, <code>_readaloud.zip, one file per
    chapter tagged with the language (and religion, as eBible's are), held
    out by book and chapter."""
    lines = [f"{book} {chapter}:{verse} {text}" for (book, chapter, verse), text in sorted(verses.items())]
    git_source(tmp_path, root, store, f"{code}_vpl.zip", {f"{code}_vpl.txt": "\n".join(lines) + "\n"},
               {"tags": [{"tags": ["verse-numbers"], "paths": ["**"]}]}, fetched)
    if twin:
        chapters = sorted({(book, chapter) for book, chapter, _ in verses})
        files = {f"{code}_002_{book}_{chapter:02d}_read.txt": f"{book} chapter {chapter}, read aloud.\n"
                 for book, chapter in chapters}
        git_source(tmp_path, root, store, f"{code}_readaloud.zip", files,
                   {"tags": [{"tags": [language, "religion"], "paths": ["**"]}], "held_out_by": READ_ALOUD_RULE},
                   fetched)


def verses(language):
    return {("GEN", chapter, verse): f"{language} words of genesis {chapter} verse {verse}"
            for chapter in range(1, 21) for verse in range(1, 9)}


def test_bible_verses_translates_held_out_chapters_only_and_scores_with_chrf(tokenizer, tmp_path, root):
    # given a model that writes the real
    # verse in language B, every verse scores chrF 100; each language comes
    # from the read-aloud twin's tags, named by the tags the other lacks; and
    # every verse asked is in a chapter the twin's held-out rule holds out
    # ("BOOK C"), as the build holds the twin's chapter files out
    store = open_store((tmp_path / "store").as_uri())
    texts = {language: verses(language) for language in ("de", "en")}
    for code, language in (("deu", "de"), ("eng", "en")):
        translation(tmp_path, root, store, code, language, texts[language])

    def answer(prompt):
        reference, language = prompt.split("Write verse ")[-1].split(", and nothing else")[0].split(" in ")
        book, place = reference.split(" ")
        chapter, verse = place.split(":")
        return said(texts[language][(book, int(chapter), int(verse))])

    context = context_of(tmp_path, tokenizer, Oracle(tokenizer, answer))
    context.store = store
    data = measure(context, "bible_verses").data
    assert data["ran"] and data["items"] == 200 and data["chrf"] == pytest.approx(100)
    assert data["languages"] == ["de+religion", "en+religion"]
    assert set(data["by_pair"]) == {"de+religion -> en+religion", "en+religion -> de+religion"}
    for item in data["scored"]:
        book, place = item["verse"].split(" ")
        assert is_held_out(f"{book} {place.split(':')[0]}", 0.5)
        chapter, verse = place.split(":")
        language = item["b"].split("+")[0]
        assert item["answer"] == {"type": "json", "value": {"answer": texts[language][(book, int(chapter),
                                                                                         int(verse))]}}


def test_the_held_out_chapters_are_the_chapters_whose_read_aloud_files_the_build_holds_out():
    # the measure and the build read one rule, eBible's read-aloud sources'
    # held_out_by: a chapter is held out when the build holds out its file,
    # in every translation at once
    module = built_in("bible_verses")
    chapters = [("GEN", n) for n in range(1, 51)] + [("PSA", n) for n in range(1, 151)]
    found = []
    for sid in ("ebible-web-07317969", "ebible-eng-web-df3ffd7c"):
        twin = real_source(sid)
        code = twin.data["address"].rsplit("/", 1)[1].removesuffix("_readaloud.zip")
        paths = {f"{code}_{number:03d}_{book}_{chapter:02d}_read.txt": f"{book} {chapter}"
                 for number, (book, chapter) in enumerate(chapters, start=2)}
        built = {path: Document(f"{sid}/{path}", sid, "0" * 64, frozenset(), held_out_group(twin, path))
                 for path in paths}
        held = module.held_out_chapters(twin, list(paths), 0.1)
        assert held == {paths[path] for path, document in built.items() if is_held_out(document.held_out_key, 0.1)}
        found.append(held)
    assert found[0] == found[1] and 0 < len(found[0]) < len(chapters)


def test_bible_verses_says_it_did_not_run_without_a_verse_source_and_its_twin_in_the_store(tokenizer, tmp_path,
                                                                                           root):
    context = context_of(tmp_path, tokenizer, Oracle(tokenizer, lambda prompt: []))
    data = measure(context, "bible_verses").data
    assert not data["ran"] and "no admitted source tagged verse-numbers" in data["reason"]

    translation(tmp_path, root, context.store, "deu", "de", verses("de"), twin=False)
    data = measure(context, "bible_verses").data
    assert not data["ran"] and data["sources_missing"] == 1
    assert "none of the 1 admitted sources tagged verse-numbers is in the store with its read-aloud twin" \
        in data["reason"]


def test_calibration_runs_as_is_on_a_pre_trained_model(trained, tmp_path, root):
    # every measure runs as is on any model: a tiny model trained one step writes through its own
    # generation cache and scores what it scores, near 0 with no block
    folder_context = open_context(trained.folder, trained.store, torch.device("cpu"))
    training = [(f"src/t{i}.md", document(f"t{i}x", 3), 2) for i in range(2)]
    held_out = [(f"src/h{i}.md", document(f"h{i}x", 3)) for i in range(2)]
    context = context_of(tmp_path / "measure", folder_context.tokenizer, folder_context.model, training, held_out)
    data = measure(context, "calibration").data
    assert data["training"]["questions"] == data["held_out"]["questions"] == 2
    for part in ("training", "held_out"):
        assert 0 <= data[part]["accuracy"] <= 1 and 0 <= data[part]["brier"] <= 1
