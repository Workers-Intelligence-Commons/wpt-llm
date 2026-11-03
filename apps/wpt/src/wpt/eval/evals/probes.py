"""Probes: how often does the model reply something alarming to a fixed question?

The questions and the patterns that flag an alarming reply are people's
text: the probe files of every admitted source (wpt_dataset.probes,
docs/evaluation.md, "Probes"), read through the repository and the
store, never the dataset. Each question is asked as one user message
through the chat template (wpt.eval.asking.replies), and the model
writes its replies in batches of at most asking.BATCH, sampled from its
plain distribution (temperature 1, every token: SAMPLING), at most
asking.MAX_TOKENS tokens each, from a generator seeded by the question's
text, so a run gives the same replies again on the same setup. A reply
is flagged when any pattern of its question matches somewhere in the
whole reply as written, its thinking and tool calls included, each
special token written as its tag (re.search on wpt_tokenizer.chat.show);
the patterns carry their own flags.

How many replies each question gets comes from one setting, the rarest
rate of alarming replies the run must catch (wpt eval's --probe-rate,
WPT_PROBE_RATE; context.probe_rate): ceil(3 / rate), the rule of three,
so a kind of reply that comes at that rate shows at least once 95 times
in 100. The result keeps every reply; the model card gives each
question's count and share flagged, and every flagged reply as written,
word for word, with the pattern that matched. Without a probe in the
repository, the test says so and returns a result that says it did not
run.
"""

import math
import re

import wpt_agent
from wpt_dataset import probes as probe_files
from wpt_dataset.repo import Repo
from wpt.eval import Result
from wpt.eval.asking import MAX_TOKENS, generator, samples
from wpt_tokenizer import chat

SAMPLING = wpt_agent.Sampling(1.0, None)  # the model's plain distribution
CAUGHT = 3  # the rule of three: replies per question = CAUGHT / rate


def replies_per_question(rate) -> int:
    """ceil(3 / rate), exactly: rate is a fractions.Fraction."""
    return math.ceil(CAUGHT / rate)


def flagged_by(text: str, patterns) -> str | None:
    """The first of patterns that matches somewhere in text, or None."""
    return next((pattern for pattern in patterns if re.search(pattern, text)), None)


def fenced(text: str) -> str:
    """text as a Markdown code block, its fence longer than any run of
    backticks in it, so it shows word for word."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def run(context):
    rate = context.probe_rate
    count = replies_per_question(rate)
    found, missing = probe_files.read(Repo.find(), context.store)
    data = {"ran": False, "rate": str(rate), "replies_per_question": count, "max_tokens": MAX_TOKENS,
            "sources_missing": missing}
    if not found:
        why = "no admitted source holds a probe file" if not missing else \
            f"the {len(missing)} admitted sources holding probe files are not in the store"
        return Result({**data, "reason": why}, f"Not run: {why}.\n")
    asked = []
    for probe in found:
        messages = [{"role": "user", "content": [{"type": "text", "text": probe.question}]}]
        written = samples(context, messages, count, generator(context, f"probes:{probe.question}"), SAMPLING)
        answers = []
        for tokens in written:
            text = chat.show(context.tokenizer, tokens)
            answers.append({"reply": chat.reply(context.tokenizer, tokens), "written": text,
                            "pattern": flagged_by(text, probe.patterns)})
        flagged = sum(answer["pattern"] is not None for answer in answers)
        asked.append({"source": probe.source, "path": probe.path, "line": probe.line, "question": probe.question,
                      "patterns": list(probe.patterns), "replies": count, "flagged": flagged,
                      "share": flagged / count, "answers": answers})
    data |= {"ran": True, "questions": asked}
    lines = [f"Each question asked {count} times (the rarest rate to catch: {rate}; a kind of reply that comes at "
             f"that rate shows at least once 95 times in 100), replies sampled at temperature 1 from every token, "
             f"at most {MAX_TOKENS} tokens each. A reply is flagged when one of its question's patterns matches "
             f"the whole reply as written, its thinking and tool calls included.", ""]
    if missing:
        lines += [f"Left out, not in the store: the probe files of {', '.join(missing)}.", ""]
    for item in asked:
        lines += [f"#### {item['source']}/{item['path']} line {item['line']}: {item['flagged']} of "
                  f"{item['replies']} replies flagged ({item['share']:.1%})", "", fenced(item["question"]), ""]
        for number, answer in enumerate((a for a in item["answers"] if a["pattern"] is not None), start=1):
            lines += [f"Flagged reply {number}, matched by the pattern:", "", fenced(answer["pattern"]), "",
                      fenced(answer["written"]), ""]
    return Result(data, "\n".join(lines))
