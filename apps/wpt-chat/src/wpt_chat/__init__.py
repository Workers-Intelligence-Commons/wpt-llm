"""Continue text with a trained model, or chat with one trained on conversations.

You type some text and press Enter; the model continues it until it ends the
document or reaches the length limit. There are no roles and no history: a
model trained on text only continues text. Empty line quits. The model writes
through its generation cache (wpt_inference), so text and conversations
may grow past its block_size. It writes by speculative decoding with the
folder's drafter (drafter/), the same text faster: a model and its
drafter are one unit, and a folder whose drafter is missing or was
trained against other weights is refused (wpt_model.load_model). The
scheduler's pass times are measured once, when the model is loaded;
--scheduler-costs RECORD_FOLDER reads instead those a run recorded
(scheduler-costs.json in that record folder of the store WPT_STORE
names, environment or .env), refused when they were measured on another
setup (the model's config, the drafter's block, the device).

--chat talks to a model trained on conversations (the answer-as-assistant
step, docs/training.md; docs/chat-format.md):
each line you type is a user message, the conversation so far goes through
the chat template, and the model writes the assistant's message until
<|im_end|>. Its thinking and tool calls are shown with their tags; tool
calls are not run.

--about prints the model folder's NOTICE: the licenses of the weights and of
generated text, and the sources the model learned from.
"""

import argparse
import io
import sys
from pathlib import Path

from wpt_inference import COSTS_FILE, costs_setup, measure_costs, next_message, replayed_costs, sample_tokens
from wpt_model import TokenizerMismatch, load_drafter, load_model, load_tokenizer
from wpt_model.notice import LICENSE, NOTICE_NAME, OUTPUT_LICENSE, REPOSITORY
from wpt_store import StoreError, store_from_settings
from wpt_tokenizer import Tokenizer, chat
from wpt_utils.device import describe, select_device

# the current shape of a conversation, wpt-chat's own constant:
# wpt_dataset.repo.CURRENT_SCHEMA["conversation"], which a test keeps equal
CONVERSATION_SHAPE = 3
MAX_NEW_TOKENS = 200


def continue_text(model, tokenizer, text, max_new_tokens=MAX_NEW_TOKENS, drafter=None, costs=None) -> str:
    """The model's continuation of text, stopping at the end-of-document
    token; with drafter, written by speculative decoding (costs: the
    scheduler's pass times, wpt_inference.measure_costs)."""
    ids = [Tokenizer.END] + tokenizer.encode(text.encode())  # END: a document starts here
    generated = []
    for next_id in sample_tokens(model, ids, max_new_tokens=max_new_tokens, drafter=drafter, costs=costs):
        if next_id == Tokenizer.END:
            break
        generated.append(next_id)
    return tokenizer.decode(generated).decode("utf-8", errors="replace")


def answer(model, tokenizer, conversation, max_new_tokens=MAX_NEW_TOKENS, drafter=None,
           costs=None) -> tuple[list[dict], str]:
    """The assistant's next message: its blocks, and its text with tags
    shown; drafter and costs as continue_text's."""
    generated = next_message(model, tokenizer, conversation, max_new_tokens=max_new_tokens, drafter=drafter,
                             costs=costs)
    return chat.reply(tokenizer, generated), chat.show(tokenizer, generated)


def licenses_line(folder) -> str:
    """What the start line says about licenses, and where the notices are."""
    notice = Path(folder) / NOTICE_NAME
    where = (f"sources and notices: {notice} (wpt-chat --about)" if notice.exists() else
             f"this folder has no {NOTICE_NAME}; notices and sources: {REPOSITORY}")
    return f"generated text: {OUTPUT_LICENSE}; weights: {LICENSE}; {where}"


def about(folder) -> str:
    notice = Path(folder) / NOTICE_NAME
    if notice.exists():
        return notice.read_text(encoding="utf-8")
    return (f"{folder} has no {NOTICE_NAME} (wpt-train writes one). WPT weights are {LICENSE} and the text they "
            f"generate is {OUTPUT_LICENSE}; the notices and the sources are at {REPOSITORY}\n")


def main(argv=None):
    # notices and generated text can hold any character; Windows pipes default to a narrower code page
    for stream in (sys.stdout, sys.stdin):
        if isinstance(stream, io.TextIOWrapper):  # not when a caller replaced it (a test, a host program)
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", default="runs/default", help="model folder (default: runs/default)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="length limit of each continuation")
    parser.add_argument("--chat", action="store_true", help="chat with a model trained on conversations (roles and history)")
    parser.add_argument("--about", action="store_true", help="print the model's licenses and sources, then quit")
    parser.add_argument("--scheduler-costs", metavar="RECORD_FOLDER",
                        help="read the drafter scheduler's pass times that a run recorded on the same setup: the name "
                             f"of its record folder in the store, which holds {COSTS_FILE} (default: measure them)")
    args = parser.parse_args(argv)
    if args.about:
        print(about(args.model), end="")
        return

    device = select_device()
    try:
        tokenizer = Tokenizer.from_bytes(load_tokenizer(args.model))
        model = load_model(args.model, device)  # refuses a folder whose drafter is missing or of other weights
        drafter = load_drafter(args.model, device)
    # ValueError: a vocabulary of another tokenizer version, or a DrafterMismatch
    except (TokenizerMismatch, ValueError) as error:
        sys.exit(f"{type(error).__name__}: {error}")
    model.eval()
    if args.scheduler_costs:  # a run's recorded pass times, on this setup only
        try:
            recorded = store_from_settings(Path.cwd() / ".env").read_record(args.scheduler_costs, COSTS_FILE)
            costs = replayed_costs(recorded, costs_setup(model.config, 1, drafter.block, device))
        except (StoreError, ValueError) as error:
            sys.exit(f"{type(error).__name__}: {error}")
    else:
        costs = measure_costs(model, 1, drafter.block)  # this machine's pass times, once
    what = "type a message" if args.chat else "type text to continue"
    print(f"model loaded ({tokenizer.vocab_size} tokens, with its drafter) on {describe(device)}. {what}, "
          "empty line quits.")
    print(licenses_line(args.model))
    conversation = {"schema": CONVERSATION_SHAPE, "messages": []}  # docs/chat-format.md
    while args.chat:
        text = input("you> ")
        if not text:
            return
        conversation["messages"].append({"role": "user", "content": [{"type": "text", "text": text}]})
        blocks, shown = answer(model, tokenizer, conversation, args.max_new_tokens, drafter, costs)
        print(f"wpt> {shown}")
        if blocks:
            conversation["messages"].append({"role": "assistant", "content": blocks})
    while True:
        text = input("> ")
        if not text:
            break
        print(text + continue_text(model, tokenizer, text, args.max_new_tokens, drafter, costs))
