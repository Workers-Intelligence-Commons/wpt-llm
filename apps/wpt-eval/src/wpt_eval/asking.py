"""What the text measures share (docs/evaluation.md, "Text measures"): a
question on a paragraph of a document, asked through the chat template and
answered by the model, and its answer graded by the check library's rules.

A question is built by the task builder's rules (wpt_dataset.builder): a
plain paragraph of a document's Markdown, its quoted start (about its
first 40%) and the rest, which the model must give. select() chooses the
documents a measure asks about, the same on every run. A question is
asked as one user message holding the prompt, which never asks for JSON:
the model's reply is text, and it is the answer (maintainer's decision,
2026-10-23; docs/chat-format.md, "The answer"). A measure that wants a
confidence then asks the agent loop's question (wpt_agent.loop.CONFIDENCE)
as one more user message, and the reply, a number from 0 to 100 and
nothing else, divided by 100, is the confidence
(wpt_agent.loop.confidence). The model writes each message through
wpt_inference.next_message, as wpt-chat --chat does, always taking its
likeliest token (wpt_agent.GREEDY; maintainer's decision, 2026-10-23), so
a run gives the same answers again on the same setup.

The measure builds the answer, as the agent loop does: the conversation's
metadata.answer is {"answer": the reply's text}, with "confidence" where
the measure asks for one, and None when the reply to the confidence
question is not such a number, which counts as no answer, as in a task.
A measure keeps it as a json block in its records (block), a shape that
can go back into sources. Every model runs the same way, a pre-trained
one included, and scores what it scores (maintainer's decision,
2026-10-23).

The answer is right when it holds the rest of the paragraph, the check
library's holds rule, and its confidence is scored with the library's
Brier rule (wpt_check, in the agent image, read through the service
layer by wpt_eval.attempts.library): one library for the tasks and the
measures.
"""

import hashlib
import heapq
import random
from dataclasses import dataclass

import torch

import wpt_agent
from wpt_agent import loop
from wpt_dataset import builder
from wpt_inference import next_message
from wpt_tokenizer import chat

from .attempts import library

SEED = 1  # every measure's generator and draws
MAX_TOKENS = 512  # tokens of one reply at most: a paragraph's rest is at most 800 bytes
CONTINUE = "Continue this passage: "


@dataclass(frozen=True)
class Question:
    document: str  # the document's id
    source: str
    markdown: bytes  # the whole document
    span: tuple[int, int]  # the paragraph's bytes
    cut: int  # where its quoted start ends (builder.quote_end)

    @property
    def quote(self) -> str:
        return _text(self.markdown[self.span[0]:self.cut])

    @property
    def rest(self) -> str:
        return _text(self.markdown[self.cut:self.span[1]])


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").strip()


def paragraphs(markdown: bytes, document: str, source: str) -> list[Question]:
    """A question on each paragraph of the Markdown that the task builder
    could ask for: plain text of the right size, with a quoted start."""
    found = []
    for span in builder.paragraphs(markdown):
        if builder.plain(markdown, span) and (cut := builder.quote_end(markdown, span)) is not None:
            found.append(Question(document, source, markdown, span, cut))
    return found


def select(documents, tokenizer, count, salt, cells, prepare=lambda question: question) -> dict:
    """{cell: [what cells[cell] gave, at most count]} of the documents
    (wpt_eval.Document, read once): for each document with a paragraph to
    ask for, one question on a paragraph drawn with SEED, prepare(question)
    made once, then for each cell, cells[cell](prepared), None when the
    document does not fit that cell. Each cell keeps the count documents
    that fit it whose id has the smallest sha256 (salted, so that two
    measures may draw other documents), the same on every run, in that
    order; a document no cell could keep is not read."""
    heaps = {cell: [] for cell in cells}  # (-key, n, value) of the count smallest keys so far, the largest first
    for n, document in enumerate(documents):
        key = int.from_bytes(hashlib.sha256(f"{salt}{document.id}".encode()).digest())
        open_ = [cell for cell, heap in heaps.items() if len(heap) < count or key < -heap[0][0]]
        if not open_:
            continue
        found = paragraphs(tokenizer.decode(document.tokens[:-1].tolist()), document.id, document.source)
        if not found:
            continue
        prepared = prepare(random.Random(f"{SEED}:{document.id}").choice(found))
        for cell in open_:
            value = cells[cell](prepared)
            if value is not None:
                heap = heaps[cell]
                (heapq.heappush if len(heap) < count else heapq.heapreplace)(heap, (-key, n, value))
    return {cell: [value for _, _, value in sorted(heap, key=lambda entry: -entry[0])] for cell, heap in heaps.items()}


def generator(context, salt="") -> torch.Generator:
    """The generator a measure's writing draws with, seeded by SEED and
    salt: with the likeliest token always taken, it decides only between
    tokens exactly as likely."""
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{salt}".encode()).digest()[:8])
    return torch.Generator(device=context.device).manual_seed(seed)


def ask(context, text, generator, max_tokens=MAX_TOKENS, confident=False) -> dict:
    """The conversation of one question: the user's message holding text,
    and the model's reply; with confident, then the confidence question
    (CONFIDENCE) and the model's reply to it. Its metadata keeps the
    answer the measure built, where the check library reads it:
    {"answer": the first reply's text}, with confident its "confidence"
    from 0 to 1, or None when the reply to the question is not a number
    from 0 to 100 (wpt_agent.loop.confidence)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    messages.append({"role": "assistant", "content": _reply(context, messages, generator, max_tokens)})
    answer = {"answer": loop.text(messages[-1]["content"])}
    if confident:
        messages.append({"role": "user", "content": [{"type": "text", "text": loop.CONFIDENCE}]})
        messages.append({"role": "assistant", "content": _reply(context, messages, generator, max_tokens)})
        sure = loop.confidence(messages[-1]["content"])
        answer = None if sure is None else {**answer, "confidence": sure}
    return {"messages": messages, "metadata": {"answer": answer}}


def _reply(context, messages, generator, max_tokens) -> list[dict]:
    """The blocks of the model's next message after messages."""
    written = next_message(context.model, context.tokenizer, {"messages": messages}, max_new_tokens=max_tokens,
                           temperature=wpt_agent.GREEDY.temperature, top_k=wpt_agent.GREEDY.top_k,
                           generator=generator)
    return chat.reply(context.tokenizer, written)


def block(conversation) -> dict:
    """The answer the measure built, as the json block its records keep."""
    return {"type": "json", "value": conversation["metadata"]["answer"]}


def continue_prompt(quote: str, before: str = "") -> str:
    """The question on a paragraph's quoted start, after the text before it when given."""
    return f"{before}{CONTINUE}{quote}"


def right(conversation, expected: str) -> bool:
    """Whether a conversation's answer holds expected (wpt_check.holds);
    false without an answer."""
    check = library(builder.IMAGE)
    return check.holds(check.answer_text(conversation), expected)


def graded(conversation, expected: str) -> dict:
    """{"right", "brier", "confidence", "answer"} of a conversation asked
    with its confidence: right as right() says; the Brier score of its
    confidence (wpt_check.brier, 1 without an answer); its confidence, None
    without one; the answer as a json block (block)."""
    found, correct = conversation["metadata"]["answer"], right(conversation, expected)
    return {"right": correct, "brier": library(builder.IMAGE).brier(conversation, correct),
            "confidence": None if found is None else found["confidence"], "answer": block(conversation)}
