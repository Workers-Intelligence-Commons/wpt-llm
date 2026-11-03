# Evaluation

`wpt eval` runs tests on a trained model and records their results in its
model card.

```sh
uv run wpt eval runs/default                      # every built-in test
uv run wpt eval runs/default memorization         # one built-in test
uv run wpt eval runs/default path/to/my_test.py   # any Python file
uv run wpt eval runs/default held-out-loss --kind-by origin   # held-out loss by kind of text
```

It reads the model folder that `wpt train` wrote, and the dataset the model
was trained on from the store named by `WPT_STORE`, checking every shard
against its recorded hash. Each result goes to `<model folder>/evals/<name>.json`
and the model card (`README.md`) is written again to show it. Training a new
model into the same folder removes the old results: they describe other
weights.

The agent measures (below) also need the repository: `wpt eval` runs from
inside it, since the measure tasks are task files of its sources, and each
task's image must be built on this machine first (`uv run wpt agent
build-image NAME`, which needs Docker), as for the `finish-tasks` step. A missing
image stops the command and names the command that builds it.

The `probes` test also reads the repository: its questions are the probe
files of its sources ("Probes", below).

`wpt scaling` fits the scaling curve of several trained models and
predicts the loss of a model before it trains ("Scaling curve", below).

## The model on a gpu worker

The tests run on the machine that runs `wpt eval`, which holds the
tokenizer, the store and the sandboxes. The model runs on a gpu worker
(the first key of `WPT_GPU_WORKERS`, else a local worker), which the
command sends the checkout and the folder's weight files; what a test asks
of the model goes through the link, as token ids, and the losses come back
as numbers
([machines.md](machines.md), "Evaluation").

`--kernels FORM` (or `WPT_KERNELS`) names what computes the model's
attention on the worker: `tilelang`, `triton` or `pytorch`, as for
`wpt train` ([model.md](model.md), "Attention"). A name that is none of
these is refused. The worker decides the form before it loads the model,
and a form that cannot work there refuses the evaluation before any test
runs. Unnamed, the worker chooses by what is installed for its GPU.

## Writing a test

A test is one Python file that defines `run`:

```python
from wpt.eval import Result

def run(context):
    documents = context.documents("held_out")
    count = sum(len(d.tokens) for d in documents)
    return Result({"held_out_tokens": count}, f"{count} held-out tokens")
```

`context` gives:

- `model`: the model, which runs on a gpu worker (below): `model.config`,
  `model.generator(seed)`, `model.writer(prompt, rows, temperature=...,
  top_k=..., stops=...)`, `model.greedy(prompts, steps)` and
  `model.losses(windows)`, the next-token loss in nats of each token of each
  window after its first, read with only the window's earlier tokens before
  it; all in token ids, and it holds no tensors on this machine;
- `device`: where the model runs, in words;
- `tokenizer`, `folder`, and `provenance` (the folder's
  `provenance.json`);
- `store`: the store holding the dataset;
- `telemetry`: the run's record, for the test's own events
  ([observability.md](observability.md));
- `documents("training")` and `documents("held_out")`: each document of that
  part once, with its `id`, `source` and `tokens` (ending with the
  end-of-document token). They are read one shard at a time as the test
  goes through them, so a dataset larger than memory works; a test that
  needs them twice calls `documents` twice, or keeps only what it needs.
- `attempts(plain=False)`: the shared run of attempts at the measure tasks
  (below), the same records for every test of one command.
- `probe_rate`: the rarest rate of alarming replies the `probes` test
  must catch (`--probe-rate`, below), a `fractions.Fraction`.
- `kind_by`: the field the `held-out-loss` test groups the held-out text
  by (`--kind-by`, below), or `None`.

Every test returns the same type, `Result`: `data`, numbers and names for
programs (anything JSON can hold), and `text`, the same findings written for
people, which the model card shows. A test is plain code, not a recipe file,
so it can measure whatever the question needs. The result file also records
the test file's sha256, so a result names the exact code that produced it.

Built-in tests live in `apps/wpt/src/wpt/eval/evals/`; a new file there
is a new built-in test.

## Memorization

A model can reproduce pieces of its training data, and those pieces keep
their source's license (see [output-notice.md](output-notice.md)). The
`memorization` test measures how often it happens:

- up to 500 documents of each part, the same on every run;
- the model reads the first 32 tokens of a document and writes 32 tokens,
  always taking its most likely token;
- a reproduction is exact when those 32 tokens equal the document's next 32.

Held-out documents give the baseline: text that is easy to predict, such as
repeated lines or license headers, is continued exactly without any
memorization. Exact reproductions of training documents well above the
baseline mean the model outputs pieces of its sources. The result lists
exact reproductions by source, so the sources whose attribution matters most
are visible.

## Held-out loss

`held-out-loss`: the model's next-token loss on every held-out token, in
nats per token, over the whole held-out part and by kind of text. Every
size of a model series shares one held-out part
([model.md](model.md), "Model series"), so the losses of its sizes are
measured on the same text.

- The held-out documents are read as one stream, each once, in shard
  order, each after the end-of-document token that ends the document
  before it, and the first after one end-of-document token.
- The stream is cut into windows of the model's `block_size` plus 1
  tokens, each window starting at the last token of the window before.
  Every token of every document is predicted exactly once, with the
  earlier tokens of its window before it.
- The gpu worker reads each window through the model's generation cache,
  4,096 tokens at a time, so the scores it holds at once are those of
  4,096 tokens, whatever the window's length.
- Each token's loss goes to its document. The result gives the loss, the
  tokens and the documents of the whole held-out part and of each source.

`--kind-by FIELD` (or `WPT_KIND_BY`) also groups the documents by one field
of their sources (`wpt/eval/kinds.py`), `tag` unless it names another, and
gives each kind's loss: the mean
over the tokens of its documents. A document of several kinds counts in
each, and a document the field gives no value is in the kind `(none)`.

- `origin`: the source's origin, the part of its id before the first `-`
  ([architecture.md](architecture.md), "The id").
- `tag`: the document's tags, as its source's tag entries give them in the
  repository, with the commits of a repository's history read as
  `code-versus-other` reads them (`wpt_dataset.tasks.document_tags`).
- `format`: the formats of the source's format entries that cover the
  document's file, in the repository.

The setting's default is `tag`, so an evaluation that sets nothing (the
one `WPT_EVALUATE=1` makes in `wpt manager`'s run included) groups by tag.

## Scaling curve

`wpt scaling` fits loss against training compute over several trained
models, and predicts the loss of a model recipe before it trains:

```sh
uv run wpt scaling run-256x5 run-384x8 runs/512x10 --predict eldest-titmouse:640x13
```

Each model is a model folder, or the name of a run whose model folder the
store keeps (`models/NAME`). Nothing is written. Each model's
`provenance.json` and `evals/held-out-loss.json` give one point:

- Compute: C = M x D. M, the training compute per token, is the one a
  series' plan counts ([model.md](model.md), "Model series"): 6 times the
  parameters one token uses plus the attention term, at the window length
  of the recipe's last text-window stage. D is the tokens its text-window steps trained on: for each
  stage, the updates done, times the windows of an update, times the
  window length. A step that continues the one before it
  ([training.md](training.md), "Continued steps") reports its updates from
  1, so it counts only the updates it adds to that step's.
- Loss: the overall loss of its `held-out-loss` evaluation (above): the
  loss per token over the full held-out text its provenance names.

A model without that evaluation is named in the output and left out of the
fit. `WPT_EVALUATE=1` in `wpt manager`'s run, or `wpt eval FOLDER
held-out-loss`, makes the evaluation. A fit needs 2 models that have it.

Every model in the fit must hold out the same text: models whose held-out
shards differ are refused, since their losses are not comparable.

The fit is least squares on the losses. With 2 or 3 points, it is a
power law, L = a x C^-b, the form of the compute scaling law of Kaplan et
al. With 4 points or more, it adds an irreducible loss E, at least 0:
L = a x C^-b + E, the power law plus a constant of Henighan et al., whose
loss is the sum of an irreducible loss and a power law in compute. The
constant takes a third number, so it waits for a fourth point to leave the
fit something to check. With 2 points, the curve passes through both, and
the command says nothing checks it. DeepSeek LLM predicts its larger
models' loss from such a power law fitted to small models.

`--predict RECIPE`, a model recipe's name or `SERIES:SIZE`, gives the loss
the fit predicts at the compute that recipe plans: the same count, from
its stages with every update done. It may repeat.

Sources:

- [Scaling Laws for Neural Language Models (Kaplan et al.)](https://arxiv.org/abs/2001.08361)
- [Scaling Laws for Autoregressive Generative Modeling (Henighan et al.)](https://arxiv.org/abs/2010.14701)
- [DeepSeek LLM: scaling laws](https://arxiv.org/abs/2401.02954)

## The shared run of attempts

The agent measures read one run of attempts at the measure tasks: the
measure-only tasks of every admitted source, whatever their tags, as
`wpt_dataset.tasks.read(..., measure=True)` gives them: those tagged
`eval`, the hash share of every task source, and the pointer tasks on
held-out text ([training.md](training.md), "Measure tasks"), read
with the model's own `held_out` share and dataset (its `provenance.json`),
once per command. The run is `wpt/eval/attempts.py`. Its attempts are
played by `wpt_agent.play`, the function the `finish-tasks` step plays its own
with: each in a fresh container of the task's image, written together by
the model alone, then scored by the task's check. The model always writes
its likeliest token (`wpt_agent.GREEDY`), so the attempts at one task
differ only in the tools they are offered, and a run gives the same
attempts again on the same setup. Checks use the `finish-tasks` step's values (300
seconds for a check, 4,000 bytes of a tool result) and seed 0.

```python
def run(context):
    records = context.attempts()            # seeded variation of the tools
    plain = context.attempts(plain=True)    # the same variants of the needs, under their real names
```

- `--attempts N` (or `WPT_ATTEMPTS`, default 4) is the number of attempts
  at each task (the option stays with greedy writing, since the tools
  offered vary from one attempt to the next).
- `plain=False`: each attempt is offered its seeded variation of the tools
  (`wpt_agent.offer`, from a generator seeded with the seed, the task and
  the attempt; [training.md](training.md), "Variation").
  `plain=True`: the same attempt, from a generator in the same state, is
  offered only the tools of the same variants of the task's needs, under
  their real names, nothing renamed, shuffled or added from the image's
  pool, with the same hidden files (`wpt_agent.needed`), so only the tools
  differ between the two runs.
- Each value of `plain` is played once per command, on the first call, and
  every later test of the same command reads the same records.
- Each record holds `task` (the `wpt_dataset.tasks.Task`), `kind` (the
  first part of the task's folder, such as `look-or-ask`), `index` (the
  attempt's number at its task, from 1), `plain`, `conversation` (the
  conversation the check reads, with its metadata `tools`, `looking`,
  `answered_by`, `answered_in` and `answer`), `score` (the check's
  `correct` part, from 0 to 1, or 0 when a gate of the task is broken;
  [training.md](training.md), "Scores") and `changes`, its
  changes to `/work`.
- Every attempt is a line of `attempts.jsonl` in the run's record folder in
  the store ([observability.md](observability.md)), with the task by its
  id, `<source id>/<task folder>`, as soon as it is scored. A scored
  attempt's line holds its conversation, its score and its changes to
  `/work`, which the rating page shows: a unified diff of each text file
  from the attempt's start to its end (`wpt_agent.changes`), a file that is
  not UTF-8 text or is larger than 64 KiB named with its size and never
  shown, a change of mode named, and the whole cut past 64 KiB with a line
  saying how much was cut.
- An attempt Docker dropped (not the model's doing) has no score and no
  record: it is printed, recorded as an `eval.attempt_dropped` event, and
  kept as a line marked `"dropped": true`, with the error and the opening
  it would have started from, as the `finish-tasks` step keeps its own.

Every measure runs as it is on any model, a pre-trained one included: an
attempt with no answer (no final reply, or no confidence from 0 to 100
after it; [chat-format.md](chat-format.md), "The answer") scores what its
check gives, 0 by the check library's rule, and no second form of the tasks
exists for base models.

A measure reads an attempt by the same rules as the checks: the check
library of the task's image (`wpt_check.py`,
[training.md](training.md), "The check library"), loaded through
the service layer by `wpt.eval.attempts.library`. Each measure gives the
mean score, and the pass rate of its attempts: an attempt passes when its
score is 1, its `correct` part at full marks with every gate of its task
holding (a broken gate scores the attempt 0). The rest of the rewards (the
calibration, the rules' parts, the tool calls' cost, the closeness of terse
attempts) weighs only in training; a measure reads the parts it needs
itself.

## Look before claim

`look-before-claim`: the share of claims naming a workspace path whose file
the attempt looked at first. Its rules:

- A claim names a file present in `/work` when the attempt starts (the
  task's files, a repository's `.git/` left out; files the attempt
  wrote do not count), by its path relative to `/work` or under it, in
  an assistant message or in the attempt's answer. One path in one
  message is one claim.
- The answer is the check library's (`metadata.answer`,
  [chat-format.md](chat-format.md), "The answer"): the final reply's
  text, read in place of the final reply's message; the confidence
  question and its reply, which follow, are not read.
- A claim counts as looked first when an earlier call of the same
  attempt that looks at sources (the check library's looking rule, from
  the conversation's `metadata.looking`) named that file itself: as its
  path argument (`wpt_check.path_of`) or in its command
  (`wpt_check.mentions`). A folder listing does not count.

## Not in the source

`not-in-the-source`: on the `evidence-removed` tasks, the pass rate of the
half whose paragraph was removed (the right answer says "not enough" and
names it) and of the half that can be answered, apart. A task is in the
removed half when `missing.txt` is next to its check.

## Ask versus look

`ask-versus-look`: on the `look-or-ask` tasks, for the half where
`NOTES.md` names the file (the task does not need `ask`) and the half
where only the reply of `ask` has it (the task needs `ask`): the pass rate
and the share of attempts that called `ask` at least once.

## Sources opened first

`sources-opened-first`: on the tasks that give sources, files under
`docs/` in `/work` or web pages under `web/` that `search` and `fetch`
answer from, the share of attempts that opened sources before answering,
by the check library's `sources_first`, and the mean number of tool calls
of an attempt, so that opening everything blindly shows.

## Robustness to tool changes

`tool-robustness`: the pass rate with plain tools (`attempts(plain=True)`:
only the needed tools under their real names) against the pass rate with
each attempt's variation of the tools, on the same tasks, and the gap,
plain minus varied. Only the tasks with scored attempts in both runs
count. The result says apart a command with no measure task and one
where Docker dropped every attempt of each task in one of the runs.

## Code versus other tasks

`code-versus-other`: the pass rates, with their counts, of code tasks
against every other task. A task is a code task when any document it points
to is tagged `code`, and for a repository pointer the tags of the files its
commit changes count (`wpt_dataset.tasks.document_tags`, through
`sources.tags_of`). A task that points to no document is another task.

## Text measures

Three built-in tests measure requirements 3 and 5 on text the model never
trained on: `calibration` (#28), `reading_gain` (#34, its text form) and
`bible_verses` (#35). They share `wpt/eval/asking.py`:

- A question is one user message through the model's chat template
  (`wpt_tokenizer.chat`). The model writes each reply through
  the chat template, as `wpt chat --chat` does, on the gpu worker
  (`context.model.writer`), always taking its likeliest token (`wpt_agent.GREEDY`), so a run gives
  the same replies again on the same setup.
- `calibration` and `reading_gain` ask for the rest of a paragraph. The
  paragraph and its quoted start follow the task builder's rules
  ([training.md](training.md), "The task builder"): plain text of
  120 to 800 bytes, its start about its first 40%. The question is
  "Continue this passage: " and the quoted start. Both choose their
  documents with one function, `asking.select`: for each document, one
  paragraph drawn with a fixed seed; of the documents that fit, those whose
  id has the smallest sha256, the same on every run.
- No prompt asks for JSON: the model's reply is text, and it is the answer
  ([chat-format.md](chat-format.md), "The answer"). The answer is right
  when it holds the rest of the paragraph: the check library's `holds`, the
  file in the agent image read through the service layer, so tasks and
  measures share one rule.
- `calibration` and `reading_gain` then ask for a confidence, as a task
  does: the agent loop's confidence question (`wpt_agent.loop.CONFIDENCE`)
  is one more user message, and the reply to it, a number from 0 to 100 and
  nothing else, divided by 100, is the confidence, which gets the Brier
  score (the library's `brier`). A reply that is not such a number counts
  as no answer: wrong, with a Brier score of 1, as a task scores it 0.
  `bible_verses` asks for no confidence.
- Each measure keeps the answer it built from the replies in its result's
  data as a `json` block, a shape that can go back into sources
  (`wpt.eval.asking.block`): `{"answer": <the reply's text>, "confidence":
  <0 to 1>}` where a confidence is asked, `null` when it is missing, and
  `{"answer": <the verse written>}` in `bible_verses`.
- Every measure runs as is on any model, a pre-trained one included: a
  model that gives no usable reply scores what it scores (near 0 for a
  pre-trained model), with no second form for base models.

`calibration`: up to 200 held-out documents and 200 training documents,
the same on every run, one question each, asked without the document.
The training half asks only about training documents the model saw more
than once, as #28 says: those the dataset holds more than once (their
rows in the training shards the model's provenance names, counted by the
shared shard reader, `wpt_store.shards.documents`). The counts are whole
only once every training shard was read, so the training shards are read
twice: once to count, once to choose. Accuracy, the Brier score and the
mean stated confidence are reported for each half apart, so a model that
doubts everything shows a low confidence on the training half too. Each
training question names how many times the dataset holds its document,
and the training half is also reported by that number.

`reading_gain`: a question on a paragraph of a held-out document is asked
twice, once after a text of the document in which the rest of the
paragraph is blanked (`[...]`), and once without any text, its confidence
asked after each reply. The text is cut around the paragraph at three
lengths, a quarter, half and all of the room the model's context leaves
for it (its `block_size` less the answer's 512 tokens and 64 for the
question), with the paragraph at the start of the text, in its middle or
at its end; the quoted start must occur only once in the text. For each
length and position, up to 50 documents long enough give the pass rate
with the text, without it, and the gain. A context too short for any text
returns a result that says the test did not run.

`bible_verses` is a translation task scored with chrF: aligned verses in
languages A and B, then the next verse in A, which the model writes in B.
Its rules:

- The verses come from eBible's numbered sources (`<code>_vpl.zip`, one
  verse per line as `BOOK C:V text`), tagged `verse-numbers`. They are read
  through the repository and the store, never through the dataset: they
  stay out of training only because no mixture selects that tag, and the
  verse alignment is used only here. Verses are aligned by their reference.
- A numbered source's language comes from its read-aloud twin, the
  read-aloud source whose address has the same eBible code
  (`<code>_readaloud.zip`): the tags of the twin's chapter files.
  Translations with the same tags are one language, and the prompt names
  each language by the tags it has and the other lacks.
- Only chapters held out by book and chapter are used, so that the same
  chapter is held out in every translation at once
  ([data-selection.md](data-selection.md), "Held-out text and
  contamination"). The rule has one owner, the twin's own `held_out_by`
  pattern: a chapter is held out when the pattern gives one of the twin's
  chapter files that chapter (`GEN 1`) as its held-out group and the group
  hashes into the model recipe's `held_out` share, the function the dataset
  build holds the file out with.
- 200 times, a pair of languages is drawn with a fixed seed, then a
  translation of each and a held-out chapter both hold at least 5 verses
  of; the prompt shows 4 verses that follow one another there, in A and B,
  then the next verse in A alone, and asks for it in B. chrF uses character
  n-grams up to 6, beta 2, spaces left out, as sacrebleu computes it for
  one sentence, from 0 to 100, against the real verse; the result gives the
  mean and each direction's.
- Without a numbered source and its twin both admitted and in the store,
  the test says so and returns a result that says it did not run.

## Reading the manual

The agent form of the reading-gain measure (`read_the_manual`): does the
model use a program better when its manual is there to read? Its tasks are
the task builder's `read-the-manual` ([training.md](training.md),
"The task builder"): each asks for a result only `ondrik` gives, a program
of the agent image written for WPT, so no training text describes it, and
half of them remove its manual from the image. `ondrik` is compiled, so the
image holds no readable documentation of it besides its manual. The tasks
are for the measures only: the builder tags their folder `eval`, so no
training step reads them, since a model trained on them would learn
`ondrik` itself and the gain would no longer measure reading.

It reads the attempts every agent measure shares (`context.attempts()`,
the measure-only tasks with the usual tool variation) and keeps those at
`read-the-manual` tasks:

- the pass rate with the manual and without it, each with its attempts,
  its passes, its tasks and its mean score;
- the gain: the pass rate with the manual minus the pass rate without it,
  unknown when a half has no attempt.

Like every measure, it runs as is on any model: an attempt with no answer
(no final reply, or no confidence from 0 to 100 after it) scores what it
scores, near 0 for a pre-trained model.

## Terse and full requests

A task can hold its request in two forms: its full `instructions`, and a
one-line form, `terse` ([training.md](training.md), "Tasks"). The
agent loop opens an attempt with either (`wpt_agent.attempts(...,
terse=True)` for the one-line form), and the task's check scores both the
same way, so the pass rates of the two forms can be compared on the same
held-out tasks. No task holds the one-line form yet, and the measure that
compares them is not written yet.

## People's ratings

People rate a model's attempts at measure tasks: pass or fail, "is this
what a co-worker would have delivered". The ratings are stored in the
repository as files of an `empty` source, so the measure is reproducible,
and they are a measure only, never a training signal: they are the only
measure that is not a reward, and no training step reads them.

A rating names the task's id, the sha256 of the model's weights, the run
record, the attempt's index, the verdict, who rated and the date. A file in
the `rating` format holds one rating per line, each a JSON object matching
`packages/wpt-dataset/src/wpt_dataset/schemas/rating.schema.json`:

```json
{"task": "<source id>/<task folder>", "weights": {"model.safetensors": "<sha256>"},
 "record": "<wpt eval run record folder>", "attempt": 1, "plain": false, "verdict": "pass",
 "by": "A. Person", "date": "2026-10-23"}
```

- `task`, `record`, `attempt` and `plain`: the attempt it rates, as the
  `attempts.jsonl` of a `wpt eval` run record holds it (the task's id, the
  run's record folder in the store, the attempt's index, and whether the
  attempt was offered only its needed tools under their real names, which
  tells apart two attempts of one task with the same index in one
  record);
- `weights`: the sha256 of each weight file of the model that made it,
  as `{file: sha256}`, as the record's `ids.json` names them
  (`inputs.weights`);
- `verdict`: `pass` or `fail`; `by`: who rated, a person's name, which
  holds no colon; `date`: the day, as `YYYY-MM-DD`.

A person rates an attempt once: their newer rating replaces their earlier
one. The data rules hold it for every way of saving: a rating file with two
ratings by one person of one attempt (the same task, record, attempt,
`plain` and `by`) fails `wpt dataset check --fetch`, and the service layer,
adding a person's rating of an attempt they rated already, replaces their
earlier line.

`wpt dataset check --fetch` checks every line against the schema and the
date against the calendar; a rating file, like a chat or task file, gives
no pre-training document. The service layer writes a rating file only in
an `empty` source, one line at a time, as the next patch, and gives a new
file its own `rating` format entry.

The "Ratings" page of `wpt web` lists the `wpt eval` runs in the store that
wrote attempts, then one run's attempts: each attempt's result (its answer
and the model's final reply, the last assistant message before the
confidence question, with every message below) and its changes to `/work`,
next to its request, and two buttons, pass and fail, that save the person's
rating into the rating file they chose (`ratings/ratings.jsonl` unless they
name another). An attempt Docker dropped is shown with its error and cannot
be rated. A last line of `attempts.jsonl` cut short by a run that stopped
while writing it is left out.

On the command line, `wpt dataset ratings ID PATH` lists what each
rating rates, who and when, and `wpt dataset save-rating ID PATH
LINE|new FILE` writes one (a person's text, marked `*`).

## Probes

The `probes` test asks a model a fixed question many times and measures
how often its reply is alarming. The questions, and the patterns that
flag an alarming reply, are people's text, stored in the repository as
files of an `empty` source, so a new question needs no code and the
measure is reproducible. Probes are a measure only: a probe file gives no
pre-training document, no dataset build keeps it, and no training step
reads it.

A file in the `probe` format holds one probe per line, each a JSON object
matching `packages/wpt-dataset/src/wpt_dataset/schemas/probe.schema.json`:

```json
{"question": "<the question>", "patterns": ["<regular expression>", "..."]}
```

- `question`: the text asked, as one user message.
- `patterns`: one or more regular expressions in Python's `re` syntax. A
  reply is flagged when any of them matches somewhere in its text
  (`re.search`). Flags, such as `(?i)` for any case, are written in the
  pattern itself: the test adds none.

`wpt dataset check --fetch` checks every line against the schema and
compiles every pattern: a pattern Python cannot compile fails the check.
The service layer writes a probe file only in an `empty` source, one line
at a time, as the next patch, and gives a new file its own `probe` format
entry.

The test reads the probes of every admitted source, in source id, file
path and line order (`wpt_dataset.probes.read`). Each question is one
user message through the model's chat template, as the text measures ask
(`wpt.eval.asking.replies`). The replies are sampled from the model's
plain distribution (temperature 1, every token), at most 512 tokens
each, written from one prompt in batches of at most 64 replies, one
after another, all drawing from one generator seeded by the question's
text, so a run gives the same replies again on the same setup.
A reply is matched as the model wrote it, its thinking and tool calls
included, each special token written as its tag
(`wpt_tokenizer.chat.show`).

One setting decides how many replies each question gets: the rarest rate
of alarming replies the run must catch, `--probe-rate RATE` (or
`WPT_PROBE_RATE`), a fraction such as `1/100` or a decimal such as
`0.01`, more than 0 and at most 1, default `1/100`. Each question is asked
ceil(3 / RATE) times, the rule of three: a kind of reply that comes at
that rate shows at least once 95 times in 100. The default gives 300
replies per question.

For each question, the result's data gives where the probe lives (its
source, file and line), the question and its patterns, the number of
replies, the number flagged and the share flagged, and every reply, as
its blocks and as written, with the pattern that flagged it or `null`.
The model card shows each question's numbers and prints each flagged
reply as written, word for word, with the pattern that matched. Without a probe in the repository, the test says it
did not run.

The "Probes" page of `wpt web` writes the probes of a probe file
(`probes/probes.jsonl` unless the person names another), each with its
question and its patterns. On the command line, `wpt dataset probes ID
PATH` lists how many patterns each probe holds, and `wpt dataset
save-probe ID PATH LINE|new FILE` writes one (a person's text, marked
`*`).
