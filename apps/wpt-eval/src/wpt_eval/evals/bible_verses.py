"""Parallel Bible verses: can the model translate the next verse, given aligned verses in two languages?

The verses come from eBible's numbered sources (one verse per line as
"BOOK C:V text", files tagged VERSE_TAG), read through the repository and
the store, not the dataset: no mixture selects that tag, so they are
never training documents. Verses are aligned by their reference. A numbered source's
language comes from its read-aloud twin, the read-aloud source whose
address has the same eBible code (https://eBible.org/Scriptures/<code>_readaloud.zip
for <code>_vpl.zip): the tags of the
twin's chapter files; translations with the same tags are one language,
and the prompt names each language by the tags it has and the other
lacks. Only chapters the held-out rule holds out are used, chosen by book
and chapter so that the same chapter is held out in every translation at
once: a chapter is held out when the
twin's own held_out_by pattern gives one of its chapter files that
chapter as its held-out group, and the group hashes into the model
recipe's held_out share, the same function the dataset build uses
(held_out_chapters), so no other translation of a verse asked for was
trained on.

The measure is translation: aligned
verse pairs in languages A and B, then the next verse in A; the model
writes it in B, and chrF scores its reply. ITEMS times, a pair of
languages A and B is drawn with SEED, then a translation of each and a
held-out chapter both hold at least SHOWN + 1 verses of; the prompt shows
SHOWN verses that follow one another there in A and B, verse by verse,
then the next verse in A alone, and asks for it in B; the prompt never
asks for JSON, and no confidence is asked. The model's reply (its text,
through the chat template, wpt_eval.asking) is scored with chrF (Popović
2015: CHRF_ORDER character n-grams, beta CHRF_BETA, spaces left out, as
sacrebleu computes it for one sentence) against the real verse, from 0 to
100, and kept with each item as the json block {"answer": the reply's
text} (wpt_eval.asking.block). Without a numbered source and its twin in
the store, the measure says so and returns a result that says it did not
run.
"""

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from wpt_dataset.build import is_held_out
from wpt_dataset.encoding import source_text
from wpt_dataset.repo import Repo
from wpt_dataset.sources import held_out_group, in_store, source_tree, tags_of
from wpt_eval import Result
from wpt_eval.asking import SEED, ask, block, generator

VERSE_TAG = "verse-numbers"
NUMBERED, READ_ALOUD = "_vpl.zip", "_readaloud.zip"  # the ends of a numbered source's address and its twin's
ITEMS, SHOWN, TRIES = 200, 4, 20
CHRF_ORDER, CHRF_BETA = 6, 2
LINE = re.compile(r"([A-Z0-9]{3} [0-9]+):([0-9]+) (.*\S)")  # "GEN 1:1 text": the chapter, the verse, its text


def chrf(hypothesis: str, reference: str) -> float:
    """chrF from 0 to 100, as sacrebleu computes it for one sentence: the
    F-score (beta CHRF_BETA, recall weighed more) of the character n-grams,
    n from 1 to CHRF_ORDER, spaces left out, their precision and recall
    each summed over the orders and divided by the number of orders both
    texts hold n-grams of."""
    hypothesis, reference = "".join(hypothesis.split()), "".join(reference.split())
    precision = recall = 0.0
    orders = 0
    for n in range(1, CHRF_ORDER + 1):
        written = Counter(hypothesis[i:i + n] for i in range(len(hypothesis) - n + 1))
        wanted = Counter(reference[i:i + n] for i in range(len(reference) - n + 1))
        matched = sum((wanted & written).values())
        precision += matched / sum(written.values()) if written else 0.0
        recall += matched / sum(wanted.values()) if wanted else 0.0
        orders += bool(written and wanted)
    if not orders or not precision + recall:
        return 0.0
    precision, recall = precision / orders, recall / orders
    return 100 * (1 + CHRF_BETA ** 2) * precision * recall / (CHRF_BETA ** 2 * precision + recall)


def held_out_chapters(twin, paths, share: float) -> set[str]:
    """The held-out chapters ("BOOK C") of a read-aloud source whose files
    are at paths: the held-out group its held_out_by gives each file
    (wpt_dataset.sources.held_out_group), kept when it hashes into share
    (wpt_dataset.build.is_held_out), as the build holds those files out."""
    return {group for path in paths if (group := held_out_group(twin, path)) is not None
            and is_held_out(group, share)}


@dataclass(frozen=True)
class Translation:
    tags: frozenset  # its twin's chapter files' tags: its language
    verses: dict  # {(chapter "BOOK C", verse): text}
    held: frozenset  # its held-out chapters


def translations(repo, store, share) -> tuple[dict, int, int]:
    """({language: {numbered source id: Translation}} of the admitted
    numbered sources whose twin the store holds too, how many admitted
    numbered sources the repository has, how many of them lack their twin
    or a copy in the store). A language is its tags joined by "+"."""
    sources = repo.sources(lambda _: None)
    twins = {source.data["address"].removesuffix(READ_ALOUD): source for source in sources
             if source.data.get("address", "").endswith(READ_ALOUD)}
    found, listed, missing = defaultdict(dict), 0, 0
    for source in sources:
        if not source.admitted or not any(VERSE_TAG in entry["tags"] for entry in source.data.get("tags", [])):
            continue
        listed += 1
        twin = twins.get(source.data["address"].removesuffix(NUMBERED))
        if twin is None or not in_store(source, store) or not in_store(twin, store):
            missing += 1
            continue
        chapters, _ = source_tree(twin, store)
        grouped = [path for path in chapters if held_out_group(twin, path) is not None]
        tags = frozenset(tag for path in grouped for tag in tags_of(twin, path))
        tree, _ = source_tree(source, store)
        verses = {}
        for path, entry in sorted(tree.items()):
            if VERSE_TAG not in tags_of(source, path):
                continue
            for line in source_text(source, path, store.get_bytes(entry.sha256)).splitlines():
                if match := LINE.fullmatch(line.strip()):
                    verses[(match[1], int(match[2]))] = match[3]
        if verses:
            found["+".join(sorted(tags))][source.id] = Translation(tags, verses,
                                                                     frozenset(held_out_chapters(twin, grouped, share)))
    return dict(found), listed, missing


def shared_chapters(a: Translation, b: Translation) -> dict:
    """{chapter: the references both translations hold there, in order} of
    the chapters both hold out, with at least SHOWN + 1 of them."""
    chapters = defaultdict(list)
    for reference in sorted(a.verses.keys() & b.verses.keys(), key=lambda r: (r[0], r[1])):
        if reference[0] in a.held and reference[0] in b.held:
            chapters[reference[0]].append(reference)
    return {chapter: references for chapter, references in sorted(chapters.items()) if len(references) > SHOWN}


def names(a: frozenset, b: frozenset) -> tuple[str, str]:
    """How the prompt names two languages: by the tags each has and the
    other lacks (all its tags when it has none of its own), joined by "+"."""
    return "+".join(sorted(a - b) or sorted(a)), "+".join(sorted(b - a) or sorted(b))


def prompt(a_name, b_name, a, b, references) -> str:
    def named(reference):
        return f"{reference[0]}:{reference[1]}"
    shown = "\n\n".join(f"{named(r)}\n{a_name}: {a[r]}\n{b_name}: {b[r]}" for r in references[:-1])
    last = references[-1]
    return (f"Verses of the Bible in two languages, {a_name} and {b_name}, each verse in both:\n\n{shown}\n\n"
            f"{named(last)}\n{a_name}: {a[last]}\n\nWrite verse {named(last)} in {b_name}, and nothing else.")


def draws(found, rng):
    """ITEMS (language A, B, their translations' ids, the references) drawn with rng."""
    languages = sorted(found)
    made, chapters = [], {}
    for _ in range(ITEMS):
        for _ in range(TRIES):
            a_key, b_key = rng.sample(languages, 2)
            a_id, b_id = rng.choice(sorted(found[a_key])), rng.choice(sorted(found[b_key]))
            if (a_id, b_id) not in chapters:
                chapters[(a_id, b_id)] = shared_chapters(found[a_key][a_id], found[b_key][b_id])
            if not chapters[(a_id, b_id)]:
                continue
            references = chapters[(a_id, b_id)][rng.choice(sorted(chapters[(a_id, b_id)]))]
            start = rng.randrange(len(references) - SHOWN)
            made.append((a_key, b_key, a_id, b_id, references[start:start + SHOWN + 1]))
            break
    return made


def run(context):
    share = context.provenance["recipe"]["data"]["held_out"]
    found, listed, missing = translations(Repo.find(), context.store, share)
    data = {"ran": False, "sources_listed": listed, "sources_missing": missing,
            "translations": sum(len(t) for t in found.values()), "languages": sorted(found)}
    if not found:
        why = (f"no admitted source tagged {VERSE_TAG} is in the repository" if not listed else
               f"none of the {listed} admitted sources tagged {VERSE_TAG} is in the store with its read-aloud twin "
               f"(wpt-dataset check --fetch fetches them)")
        return Result({**data, "reason": why}, f"Not run: {why}.\n")
    if len(found) < 2:
        why = f"the {VERSE_TAG} sources in the store hold one language only ({', '.join(found)})"
        return Result({**data, "reason": why}, f"Not run: {why}.\n")
    items = draws(found, random.Random(f"bible-verses:{SEED}"))
    if not items:
        why = "no two translations of different languages share a held-out chapter"
        return Result({**data, "reason": why}, f"Not run: {why}.\n")
    drawn, scored = generator(context, "bible-verses"), []
    for a_key, b_key, a_id, b_id, references in items:
        a, b = found[a_key][a_id], found[b_key][b_id]
        a_name, b_name = names(a.tags, b.tags)
        conversation = ask(context, prompt(a_name, b_name, a.verses, b.verses, references), drawn)
        last = references[-1]
        scored.append({"a": a_key, "b": b_key, "a_source": a_id, "b_source": b_id, "verse": f"{last[0]}:{last[1]}",
                       "chrf": chrf(conversation["metadata"]["answer"]["answer"], b.verses[last]),
                       "answer": block(conversation)})
    by_pair = defaultdict(list)
    for item in scored:
        by_pair[f"{item['a']} -> {item['b']}"].append(item["chrf"])
    data |= {"ran": True, "items": len(scored), "chrf": sum(i["chrf"] for i in scored) / len(scored),
             "by_pair": {pair: {"items": len(values), "chrf": sum(values) / len(values)}
                         for pair, values in sorted(by_pair.items())},
             "scored": scored}
    lines = [f"{len(scored)} verses of held-out chapters translated from language A to B after {SHOWN} aligned "
             f"verses; chrF from 0 to 100 against the real verse, over {data['translations']} translations in "
             f"{len(found)} languages.", "", f"- mean chrF: {data['chrf']:.1f}"]
    lines += [f"- {pair}: {got['items']} verses, chrF {got['chrf']:.1f}" for pair, got in data["by_pair"].items()]
    return Result(data, "\n".join(lines) + "\n")
