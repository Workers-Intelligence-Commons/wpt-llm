"""Reading gain: does the model answer better with the document in front of it?

A question on a paragraph of a held-out document (wpt_eval.asking: the task
builder's paragraph and quote rules) is asked twice: once after a text of
the document in which the rest of that paragraph is blanked (BLANK), and
once without any text, as the calibration test asks it, its confidence
asked after the reply. The model's reply, text, is its answer, right when
it holds the rest of the paragraph (the check library's holds) and a
confidence from 0 to 100 came after it, and each cell keeps both answers
as json blocks (wpt_eval.asking.block). The given text is cut from the
document around the paragraph at each length of LENGTHS (shares of the
room the model's context leaves for it: its block_size less the answer's
MAX_TOKENS and PROMPT_TOKENS for the question's own words) and each
position of POSITIONS: the paragraph at the start of the text, in its
middle, or at its end. For each length and position, up to QUESTIONS
documents long enough on the sides that position needs, those whose id has
the smallest sha256 (asking.select), and whose quoted start occurs only
once in the given text. The result gives the pass rate with and without
the text, and the gain, for each length and position.
"""

from dataclasses import dataclass

from wpt_dataset import builder
from wpt_eval import Result
from wpt_eval.asking import MAX_TOKENS, Question, ask, block, continue_prompt, generator, right, select

QUESTIONS = 50  # per length and position
LENGTHS = (0.25, 0.5, 1.0)  # shares of the room the context leaves for the given text
POSITIONS = ("start", "middle", "end")
PROMPT_TOKENS = 64  # room kept for the question's words around the given text
BLANK = " [...]"
LEFT_OUT = "\n\nIn the text above, the rest of one paragraph is left out, marked [...]. "


@dataclass(frozen=True)
class Read:
    """A question and its document's tokens around the paragraph, read once."""
    question: Question
    before: list  # the tokens of the Markdown before the paragraph
    after: list  # the tokens after it
    blanked: bytes  # the paragraph's quoted start and BLANK
    blanked_tokens: int


def read(tokenizer, question) -> Read:
    markdown, (start, end) = question.markdown, question.span
    blanked = markdown[start:question.cut] + BLANK.encode()
    return Read(question, tokenizer.encode(markdown[:start]), tokenizer.encode(markdown[end:]), blanked,
                len(tokenizer.encode(blanked)))


def given_text(found: Read, tokenizer, length, position) -> tuple[Question, str] | None:
    """(the question, the text of the document around its paragraph, about
    length tokens, the rest of the paragraph blanked and the paragraph at
    position); None when the document is too short on the side it needs, or
    the quoted start occurs elsewhere in the text."""
    question = found.question
    room = length - found.blanked_tokens
    if room < 0:
        return None
    before, after = found.before, found.after
    taken = {"start": (0, room), "middle": (room // 2, room - room // 2), "end": (room, 0)}[position]
    if len(before) < taken[0] or len(after) < taken[1]:
        return None
    text = b"".join([tokenizer.decode(before[len(before) - taken[0]:]), found.blanked,
                     tokenizer.decode(after[:taken[1]])])
    if builder.occurs(builder.words(question.markdown[question.span[0]:question.cut]), [text]) != 1:
        return None
    return question, text.decode("utf-8", errors="ignore")  # a cut inside a character's bytes leaves it out


def chosen(context, lengths) -> dict:
    """{(length, position): [(question, given text)]}: for each, up to
    QUESTIONS held-out documents whose id has the smallest sha256 and whose
    drawn paragraph fits."""
    tokenizer = context.tokenizer
    cells = {(length, position): (lambda found, length=length, position=position:
                                  given_text(found, tokenizer, length, position))
             for length in lengths for position in POSITIONS}
    return select(context.documents("held_out"), tokenizer, QUESTIONS, "reading-gain", cells,
                  lambda question: read(tokenizer, question))


def run(context):
    room = context.model.config.block_size - MAX_TOKENS - PROMPT_TOKENS
    lengths = {share: int(share * room) for share in LENGTHS}
    data = {"context_tokens": context.model.config.block_size, "questions_per_cell": QUESTIONS, "cells": []}
    lines = [f"A paragraph's quoted start asked for with a text of the held-out document around it (the rest of "
             f"the paragraph blanked) and without; right when the answer holds the rest. Context "
             f"{context.model.config.block_size} tokens.", ""]
    if room <= 0:
        lines.append(f"- not run: the context leaves no room for a text beside the answer's {MAX_TOKENS} tokens")
        return Result({**data, "ran": False}, "\n".join(lines) + "\n")
    cells = chosen(context, sorted(set(lengths.values())))
    drawn = generator(context, "reading-gain")
    for share, length in lengths.items():
        for position in POSITIONS:
            asked = cells[(length, position)]
            answers = [(q, ask(context, continue_prompt(q.quote, f"{text}{LEFT_OUT}"), drawn, confident=True),
                        ask(context, continue_prompt(q.quote), drawn, confident=True)) for q, text in asked]
            with_text = sum(right(given, q.rest) for q, given, _ in answers)
            without = sum(right(alone, q.rest) for q, _, alone in answers)
            count = len(asked)
            cell = {"share": share, "tokens": length, "position": position, "questions": count,
                    "with_text": with_text / count if count else None, "without": without / count if count else None,
                    "gain": (with_text - without) / count if count else None,
                    "documents": [q.document for q, _ in asked],
                    "answers": [{"document": q.document, "with_text": block(given), "without": block(alone)}
                                for q, given, alone in answers]}
            data["cells"].append(cell)
            where = f"- {share:.0%} of the room ({length} tokens), paragraph at the {position}"
            lines.append(f"{where}: no held-out document is long enough" if not count else
                         f"{where}: {count} questions, {cell['with_text']:.1%} right with the text, "
                         f"{cell['without']:.1%} without, gain {cell['gain']:+.1%}")
    return Result({**data, "ran": True}, "\n".join(lines) + "\n")
