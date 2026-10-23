"""Confidence calibration: does the model's stated confidence match how often it is right?

For up to QUESTIONS documents of each part, held-out (never trained on)
and training, a question on one paragraph (wpt_eval.asking: the task
builder's paragraph and quote rules, the documents chosen by
asking.select) is asked without the document: the paragraph's quoted
start after "Continue this passage: ". The model's reply, text, is its
answer; then the agent loop's confidence question is asked, and the
reply, a number from 0 to 100, gives the confidence (wpt_eval.asking.ask
with confident). An answer is right when it holds the rest of the
paragraph (the check library's holds), and its confidence is scored with
the Brier score (the library's brier: 1 without an answer, as when the
reply to the question is not such a number). Each question keeps its
answer as a json block, {"answer": ..., "confidence": ...} or null.
Accuracy and the Brier score are reported for each part apart: a model
that doubts everything shows a low confidence on the training half too.

The training half asks only about training documents seen several times,
as issue #28 says: those the dataset holds more than once (their passes,
the rows of the training shards the model's provenance names, counted by
the shared shard reader). The counts are whole only once every training
document was read, so the training shards are read twice: once to count,
once to choose. The training half is also reported by that number.
"""

from collections import Counter

from wpt_eval import Result
from wpt_eval.asking import ask, continue_prompt, generator, graded, select

QUESTIONS = 200  # per part
PARTS = {"held_out": "held-out documents", "training": "training documents seen more than once"}


def questions(documents, tokenizer):
    """Up to QUESTIONS questions on documents, one each (asking.select)."""
    return select(documents, tokenizer, QUESTIONS, "calibration", {None: lambda question: question})[None]


def summary(graded_questions) -> dict:
    count = len(graded_questions)
    answered = [g for g in graded_questions if g["confidence"] is not None]
    return {"questions": count, "answered": len(answered),
            "accuracy": sum(g["right"] for g in graded_questions) / count if count else None,
            "brier": sum(g["brier"] for g in graded_questions) / count if count else None,
            "mean_confidence": sum(g["confidence"] for g in answered) / len(answered) if answered else None}


def run(context):
    passes = Counter()
    for _ in context.documents("training", passes):  # every training document's passes, before any is chosen
        pass
    asked = {"held_out": questions(context.documents("held_out"), context.tokenizer),
             "training": questions((document for document in context.documents("training")
                                    if passes[document.id] > 1), context.tokenizer)}
    data = {"questions_per_part": QUESTIONS, "questions": []}
    lines = ["Questions without the document: the start of a paragraph, \"continue this passage\", then the "
             "confidence asked from 0 to 100; right when the answer holds the rest of the paragraph. Brier score of "
             "the stated confidence (0 best, 1 worst, 1 without a confidence from 0 to 100).", ""]
    for part, label in PARTS.items():
        drawn = generator(context, f"calibration:{part}")
        found = []
        for question in asked[part]:
            conversation = ask(context, continue_prompt(question.quote), drawn, confident=True)
            found.append({"part": part, "document": question.document, "source": question.source,
                          **({"passes": passes[question.document]} if part == "training" else {}),
                          **graded(conversation, question.rest)})
        data["questions"] += found
        data[part] = summary(found)
        if part == "training":
            data[part]["by_passes"] = {str(n): summary([g for g in found if g["passes"] == n])
                                       for n in sorted({g["passes"] for g in found})}
        if not found:
            lines.append(f"- {label}: none holds a paragraph to ask for")
            continue
        got = data[part]
        confidence = "no confidence given" if got["mean_confidence"] is None else \
            f"mean confidence {got['mean_confidence']:.2f}"
        lines.append(f"- {label}: {got['questions']} questions, accuracy {got['accuracy']:.1%}, Brier score "
                     f"{got['brier']:.3f}; {got['answered']} gave a confidence from 0 to 100, {confidence}")
    for n, got in data["training"].get("by_passes", {}).items():
        lines.append(f"  - training documents the dataset holds {n} times: {got['questions']} questions, "
                     f"accuracy {got['accuracy']:.1%}, Brier score {got['brier']:.3f}")
    return Result(data, "\n".join(lines) + "\n")
