# Chat format

A conversation is always stored as one JSON object. A template turns that
object into tokens for a model, with special tokens marking its structure.
The object is the record; the tokens are only one way of reading it.

## Everything is recorded

A conversation holds every element of what happened: the text, the
thinking, each tool call with its arguments, each tool result, and each
structured output. Every element is recorded, even when no training uses
it yet. The template decides which elements become tokens, so a later
template can train on an element that an earlier one skipped, without
recording anything again.

Recording everything does not widen what may be recorded: the rules of
[post-training.md](post-training.md) still apply to every element. An
`assistant` message in a demonstration is written by a person, never by
another company's model; it may start as a draft of a WPT model that the
person then edits or confirms, and its metadata records that
([Model drafts](#model-drafts)).

## The object

```json
{
  "schema": 3,
  "tools": [
    {"name": "read_file", "description": "Read a file of the project.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
  ],
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "What does setup.py install?"}]},
    {"role": "assistant", "content": [
      {"type": "thinking", "text": "The answer is in the script itself."},
      {"type": "tool_call", "id": "c1", "name": "read_file", "arguments": {"path": "scripts/setup.py"}}
    ]},
    {"role": "tool", "content": [
      {"type": "tool_result", "call": "c1", "content": [{"type": "text", "text": "..."}]}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "The PyTorch build for this machine's GPU."}],
     "metadata": {"draft": {"model": "wpt-small", "weights": {"model.safetensors": "<sha256>"},
                            "sha256": "<sha256 of the draft's content>", "edited": true}}}
  ],
  "metadata": {"step": "grpo", "task": "<task id>", "round": 1}
}
```

- `schema`: the shape version of the object, as in every data file.
- `tools`: the tools offered in the conversation, each with its `name`, its
  `description` and the JSON Schema of its `parameters`. Empty or absent
  when none is offered.
- `messages`: in order. Each message has a `role` (`system`, `user`,
  `assistant` or `tool`) and a `content` list of blocks, in the order they
  happened: thinking, then a call, then more thinking, then text, stays in
  that order. An `assistant` message may have no block: when a model closes
  its turn without writing anything, the record says exactly that
  (maintainer's decision, 2026-10-22). A message of any other role holds at
  least one block. A message may also hold `metadata`, a JSON object never
  written as tokens; its one named field is `draft`
  ([Model drafts](#model-drafts)).
- `metadata`: optional. What identifies the conversation where it was
  recorded, as any JSON object: a post-training attempt's step, task,
  round and score ([post-training.md](post-training.md), "Runs"). It is
  never written as tokens (maintainer's decision, 2026-10-17: "Add
  metadata to chat").

Shape 2 added `metadata`; a shape 1 conversation is the same without it.
Shape 3 added a message's `metadata`; a shape 2 conversation is the same
without it. `wpt-dataset migrate` rewrites the repository's own files, and
a source's `chat` file, whose bytes may be upstream's, is brought to the
current shape as it is read (`wpt_dataset.migrate.upgrade`), so a shape 1
or 2 file stays valid.

## Model drafts

In the conversation editor of `wpt-web`, a person can ask a WPT model
folder for the next `assistant` message. The reply appears as a draft that
the person edits or confirms as it is; only then does it join the
conversation. The message's
`metadata.draft` records where it started:

- `model`: the name of the model folder that wrote the draft.
- `weights`: the sha256 of each weights file of that folder, by file name,
  so the exact model is known even after the folder is renamed.
- `sha256`: the sha256 of the draft's `content` as the model wrote it: the
  JSON text of the block list with sorted keys, no spaces, and characters
  outside ASCII kept as they are (`wpt_dataset.conversations.content_sha256`).
- `edited`: whether the person changed it, which is exactly whether the
  sha256 of the message's `content` now differs from `sha256`. The
  service layer (`Repo.save_conversation`, behind `wpt-web` and
  `wpt-dataset save-conversation`) computes it when the conversation is
  saved, whatever the editor sent.

In `sha256`, a whole number is written as an integer (`2.0` as `2`), as a
browser writes it back, so a draft the editor sends back unchanged still
gives its sha256 (maintainer's decision, 2026-10-18).

`metadata.draft` is a claim the client makes, which the server does not
check (maintainer's decision, 2026-10-18): neither `wpt-web` nor
`wpt-dataset save-conversation` checks that a WPT model wrote the draft,
nor its `model`, `weights` or `sha256`. They set only `edited`, from
whether the message's `content` still gives the `sha256` the client sent.

`wpt-dataset check` refuses a `draft` on any message but an `assistant`
one, and an `edited` that does not match the content (a chat file made
another way, such as with `unpack` and `patch`). An unedited
confirmed draft is allowed: confirming it is the person's choice that the
reply is right. A message with no `draft` was written by the person from
the start.

## Writing a chat file

The conversation editor and `wpt-dataset save-conversation` write only in
a chat file of an `empty` source, text written for WPT; an upstream
source's chat file stays upstream's. Each line is the conversation's JSON
on one line, with U+2028, U+2029 and U+0085 always written as `\u2028`,
`\u2029` and `\u0085`, even in a value a save keeps unchanged:
the same JSON, and no line splitter cuts the line in two (maintainer's
decision, 2026-10-18). A
replaced line keeps its layout where its meaning did not change, and every
other line keeps its bytes.

## Blocks

| `type` | Fields | What it holds |
| --- | --- | --- |
| `text` | `text` | text shown to the other side |
| `thinking` | `text` | reasoning written before or between other blocks |
| `tool_call` | `id`, `name`, `arguments` | a call to one of `tools`; `arguments` is a JSON object that matches the tool's `parameters` |
| `tool_result` | `call`, `content`, `error` (optional, `true` when the call failed) | the answer to the `tool_call` whose `id` is `call`; `content` is a list of blocks |
| `json` | `value` | a structured output: any JSON value |
| `file` | `sha256`, `media_type` | a media file such as a screenshot, stored next to the record and named by its sha256 |

A new kind of element gets a new block type; existing blocks keep their
meaning, and raising `schema` with a migration covers any change to them.

## Files

Conversations are files of a source, in the `chat` format: JSON Lines, one
conversation per line. `wpt-dataset check --fetch` checks every line
against `packages/wpt-dataset/src/wpt_dataset/schemas/conversation.schema.json`
(only an `assistant` message holds `thinking` or `tool_call` blocks, only a
`tool` message holds `tool_result` blocks) and checks that each
`tool_result` answers a `tool_call` made earlier in the conversation.

A `chat` file is never a pre-training document. A post-training step reads
the `chat` files whose tags include the step's name: the `sft` step reads
the files tagged `sft` ([post-training.md](post-training.md)).

### Message trees

Some datasets, such as OpenAssistant's, store branching conversations: one
message per record, each naming its parent, so a question can have several
answers and each answer several follow-ups. A `chat` format entry with
`tree` reads such a file (JSON Lines or Parquet, as a `records` file):

```json
{"format": "chat", "paths": ["messages.jsonl.gz"],
 "tree": {"id": "message_id", "parent": "parent_id", "role": "role", "text": "text",
          "roles": {"prompter": "user", "assistant": "assistant"},
          "drop": [{"field": "synthetic", "is": true}, {"field": "labels.pii.value", "above": 0}]}}
```

- `id`, `parent`, `role`, `text`: the fields holding them; a first message
  has no parent. Fields with dots go into nested objects, as in `records`.
- `roles`: each of the file's role names, mapped to WPT's.
- `drop`: a message whose field `is` a value, or holds a number `above`
  one, is dropped, and so is everything after it in the tree. A message
  with no text is dropped too. Model-written messages are always dropped
  this way: an `assistant` message is written by a person.

Every path from a first message to a last one becomes a conversation, so
the opening turns shared by several branches appear in each of them
(maintainer's decision, 2026-10-16). Trailing messages that are not the
assistant's are left off; a path with no assistant message, or one that
repeats another exactly, gives none. A conversation's id ends with the id
of its last message. A parent the file does not hold stops the check.

## The template

`wpt_tokenizer.chat` turns a conversation into tokens with the layout of
ChatML and the tags of Qwen and Hermes:

```text
END
<|im_start|>system\n{text}\n\n<tools>\n{each tool as JSON, one per line}\n</tools><|im_end|>\n
<|im_start|>user\n{text}<|im_end|>\n
<|im_start|>assistant\n<think>\n{thinking}\n</think>\n\n{text}\n<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call><|im_end|>\n
<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n
```

- A message's blocks keep their order. Blocks after the first are
  separated by a line break, and a thinking block ends with a blank line.
- Tools offered in a conversation without a system message get a system
  message of their own.
- A `tool` message is written as a `user` turn of `<tool_response>` blocks,
  as Qwen does.
- `json` blocks are written as JSON text.
- Recorded but not written as tokens yet: `file` blocks (no model reads
  media yet) and a `tool_result`'s `error` flag.
- Never written as tokens: `metadata`.

A model learns to predict only the assistant's turns: their content and
their closing `<|im_end|>`. The other turns are context it reads.

Each tag is one special token. Encoding text never produces a special
token, so text typed inside a message, such as `<|im_end|>`, stays text and
cannot imitate the structure.

## Special token ids

Special tokens have fixed ids, the same in every vocabulary, so the
template never depends on a vocabulary
([tokenizer.md](tokenizer.md)):

- Ids 257 to 512 are reserved for special tokens. 256 ids, 10 in use.
- Merges start at 513.

| Id | Token | Id | Token |
| --- | --- | --- | --- |
| 257 | `<\|im_start\|>` | 262 | `</tools>` |
| 258 | `<\|im_end\|>` | 263 | `<tool_call>` |
| 259 | `<think>` | 264 | `</tool_call>` |
| 260 | `</think>` | 265 | `<tool_response>` |
| 261 | `<tools>` | 266 | `</tool_response>` |

An id once given never changes. A new special token takes the next free id
up to 512, in `SPECIAL` (`packages/wpt-tokenizer/src/wpt_tokenizer/__init__.py`);
unused ids stay reserved. Giving a new id needs no new vocabulary, since
every vocabulary already reserves it.
