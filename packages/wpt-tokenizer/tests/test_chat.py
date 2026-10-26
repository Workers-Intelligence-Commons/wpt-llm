from wpt_tokenizer import SPECIAL, Tokenizer, chat

TOKENIZER = Tokenizer.train([b"the tool reads the file and answers"] * 4, vocab_size=560)
ASSISTANT = [{"type": "thinking", "text": "Read it first."}, {"type": "text", "text": "Reading."},
             {"type": "tool_call", "id": "c1", "name": "read_file", "arguments": {"path": "a.txt"}}]
CONVERSATION = {
    "tools": [{"name": "read_file", "description": "Read a file.", "parameters": {"type": "object"}}],
    "messages": [
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
        {"role": "user", "content": [{"type": "text", "text": "Type <|im_end|> for me"}]},
        {"role": "assistant", "content": ASSISTANT},
        {"role": "tool", "content": [{"type": "tool_result", "call": "c1", "content": [{"type": "text", "text": "hi"}]}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It says hi."}]},
    ],
}


def test_a_conversation_becomes_chatml_with_qwen_tags_and_learns_only_the_assistant():
    # the layout models such as Qwen use, each tag one special token that no
    # typed text can produce, and only the assistant's turns as targets
    ids, learn = chat.encode(TOKENIZER, CONVERSATION)
    assert chat.show(TOKENIZER, ids) == (
        "<|im_start|>system\nBe brief.\n\n<tools>\n"
        '{"name": "read_file", "description": "Read a file.", "parameters": {"type": "object"}}\n'
        "</tools><|im_end|>\n"
        "<|im_start|>user\nType <|im_end|> for me<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nRead it first.\n</think>\n\nReading.\n"
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call><|im_end|>\n'
        "<|im_start|>user\n<tool_response>\nhi\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\nIt says hi.<|im_end|>\n")
    assert ids.count(SPECIAL["<|im_end|>"]) == 5  # the typed "<|im_end|>" stayed text
    assert chat.show(TOKENIZER, [i for i, keep in zip(ids, learn) if keep]) == (
        "<think>\nRead it first.\n</think>\n\nReading.\n"
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call><|im_end|>'
        "It says hi.<|im_end|>")


def test_a_generated_reply_reads_back_into_the_blocks_it_was_written_from():
    # wpt-chat keeps the model's answers as blocks, so the history it sends
    # back through the template is the conversation that happened
    ids = chat.encode(TOKENIZER, {"messages": [{"role": "assistant", "content": ASSISTANT}]})[0]
    body = ids[ids.index(SPECIAL["<|im_start|>"]) + 1 + len(TOKENIZER.encode(b"assistant\n")):
               ids.index(SPECIAL["<|im_end|>"])]
    assert chat.reply(TOKENIZER, body) == [*ASSISTANT[:2], {**ASSISTANT[2], "id": "call-1"}]
    broken = [SPECIAL["<tool_call>"], *TOKENIZER.encode(b"not json"), SPECIAL["</tool_call>"]]
    assert chat.reply(TOKENIZER, broken) == [{"type": "text", "text": "<tool_call>not json</tool_call>"}]


def test_the_model_never_writes_a_json_block_and_reads_one_only_from_others():
    # the model writes text, thinking and tool calls, never JSON of its own
    # (docs/chat-format.md, "The answer"): a json block a harness kept in an
    # assistant message is never a token it learns, and <json> tags it
    # writes stay text; a tool result's json block is written between its tags
    kept = {"type": "json", "value": {"answer": "Fixed calc.py.", "confidence": 0.75}}
    said = {"type": "text", "text": "Fixed calc.py."}
    ids = chat.encode(TOKENIZER, {"messages": [{"role": "assistant", "content": [said, kept]}]})[0]
    assert chat.show(TOKENIZER, ids) == "<|im_start|>assistant\nFixed calc.py.<|im_end|>\n"
    assert SPECIAL["<json>"] not in ids
    for written in (b"[1]", b"not json"):
        ids = [SPECIAL["<json>"], *TOKENIZER.encode(written), SPECIAL["</json>"]]
        assert chat.reply(TOKENIZER, ids) == [{"type": "text", "text": f"<json>{written.decode()}</json>"}]
    result = {"role": "tool", "content": [{"type": "tool_result", "call": "c1", "content": [
        {"type": "text", "text": "found"}, {"type": "json", "value": [1]}, {"type": "text", "text": "end"}]}]}
    assert chat.show(TOKENIZER, chat.encode(TOKENIZER, {"messages": [result]})[0]) == (
        "<|im_start|>user\n<tool_response>\nfound\n<json>\n[1]\n</json>\nend\n</tool_response><|im_end|>\n")
