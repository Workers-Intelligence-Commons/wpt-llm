"""Continue text with a trained model.

You type some text and press Enter; the model continues it until it ends the
document or reaches the length limit. There are no roles and no history: a
pre-trained model only continues text. Empty line quits.

--about prints the model folder's NOTICE: the licenses of the weights and of
generated text, and the sources the model learned from.
"""

import argparse
import io
import sys
from pathlib import Path

from wpt_device import describe, select_device
from wpt_inference import sample_tokens
from wpt_model import TokenizerMismatch, load_model, load_tokenizer
from wpt_model.notice import LICENSE, NOTICE_NAME, OUTPUT_LICENSE, REPOSITORY
from wpt_tokenizer import Tokenizer

MAX_NEW_TOKENS = 200


def continue_text(model, tokenizer, text, max_new_tokens=MAX_NEW_TOKENS) -> str:
    """The model's continuation of text, stopping at the end-of-document token."""
    ids = [Tokenizer.END] + tokenizer.encode(text.encode())  # END: a document starts here
    generated = []
    for next_id in sample_tokens(model, ids, context_size=model.config.block_size, max_new_tokens=max_new_tokens):
        if next_id == Tokenizer.END:
            break
        generated.append(next_id)
    return tokenizer.decode(generated).decode("utf-8", errors="replace")


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
    parser.add_argument("--about", action="store_true", help="print the model's licenses and sources, then quit")
    args = parser.parse_args(argv)
    if args.about:
        print(about(args.model), end="")
        return

    device = select_device()
    try:
        tokenizer = Tokenizer.from_bytes(load_tokenizer(args.model))
    except TokenizerMismatch as error:
        sys.exit(f"TokenizerMismatch: {error}")
    model = load_model(args.model, device)
    model.eval()
    print(f"model loaded ({tokenizer.vocab_size} tokens) on {describe(device)}. type text to continue, empty line quits.")
    print(licenses_line(args.model))
    while True:
        text = input("> ")
        if not text:
            break
        print(text + continue_text(model, tokenizer, text, args.max_new_tokens))
