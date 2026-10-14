"""Continue text with a trained model.

You type some text and press Enter; the model continues it until it ends the
document or reaches the length limit. There are no roles and no history: a
pre-trained model only continues text. Empty line quits.
"""

import argparse
from pathlib import Path

from wpt_device import describe, select_device
from wpt_inference import sample_tokens
from wpt_model import load_model
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", default="runs/default", help="model folder (default: runs/default)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="length limit of each continuation")
    args = parser.parse_args()

    device = select_device()
    tokenizer = Tokenizer.load(Path(args.model) / "tokenizer.json")
    model = load_model(args.model, device)
    model.eval()
    print(f"model loaded ({tokenizer.vocab_size} tokens) on {describe(device)}. type text to continue, empty line quits.")
    while True:
        text = input("> ")
        if not text:
            break
        print(text + continue_text(model, tokenizer, text, args.max_new_tokens))
