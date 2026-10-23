"""Autoregressive sampling: feed the tokens so far, pick the next one, repeat.

A Writer writes the next tokens of one or several rows (attempts) that
start from one prompt. With a model that keeps a generation cache
(wpt_model's WPT: new_cache, extend, undo; docs/model.md, "Generation"),
the prompt is read once and copied into every row, each row's tokens are
read once, and the rows are read together in one batch; there is no
length limit. Any other model that maps (B, T) token ids to (B, T, vocab)
scores works too: it reads each row's whole text again for every token.

With a drafter (speculative decoding, model with a cache only), each step
the drafter proposes up to `draft` tokens per row, the model reads them in
one pass, and each is kept or rejected by the rule that leaves the model's
own distribution unchanged (accepted with probability min(1, p/q), a
rejected one replaced by a draw from max(0, p - q), normalized; when all
are kept, one more token from the model); the cache then undoes the
rejected ones. A drafter is any object with

    features   the model's layers whose hidden states it reads (may be empty)
    window     how many of the last read tokens' hidden states it reads
    draft(model, histories, context, count, probabilities, generator)
        -> (tokens (R, count), their distributions (R, count, vocab), confidences (R, count) or None)

that proposes count tokens after each of the R token lists (each ending
with its anchor, the last token written, which the model has not read),
each drawn from the distribution it gives back. probabilities turns scores
into the distribution the writer samples from (its temperature and top_k),
which a drafter may use for its own. context is (features (R, window, F),
lengths (R,)): the hidden states of each row's last window tokens read,
side by side (zeros before the text), and how many tokens each row has
read; None when features is empty. wpt_model.Drafter is WPT's (DSpark). A
row whose pending tokens are more than its anchor (a tool result) has them
read first, so that the drafter's context reaches its anchor.

With confidences and costs, the writer verifies only the drafts worth
their cost (DSpark's Hardware-Aware Prefix Scheduler, schedule): a draft's
chance to be kept is the product of the confidences up to it, and drafts
are admitted while the expected tokens per second grow. costs gives the
seconds of one pass that reads n tokens per row, n = 1 .. draft + 1, for
the writer's number of rows: a run measures them once (measure_costs),
writes them into its record with the setup they were measured on
(costs_setup, costs_file) and may replay a table an earlier run recorded
on the same setup (replayed_costs), since the verified drafts, and with
them the tokens a seeded generator draws, follow these times (never
their distribution). Without confidences or costs, every draft is
verified.

outcomes, when given a list, receives one entry per row and speculative
step: (the confidences of the drafts the model verified, how many of them
it kept), what a drafter's calibration learns from (wpt_inference.drafting).

The writer has no idea what the tokens mean; the caller says which tokens
end a turn (stops) and how many tokens each row may write. next_message
is the one exception, shared by wpt-chat --chat and the text measures: the
tokens of the assistant's next message, through the chat template.
"""

import json
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from wpt_tokenizer import END, chat
from wpt_utils.device import processor_name

DRAFT = 5  # draft tokens per row and step: the most DSpark's production drafter proposes (docs/model.md)
COSTS_FILE = "scheduler-costs.json"  # a run's measured costs, in its record (costs_file, replayed_costs)


class Writer:
    """Writes for `rows` rows that all start from prompt (a list of ints, at
    least one). temperature: >1 = more inventive, <1 = more predictable.
    top_k: sample only among the k highest-scoring tokens (None = all).
    generator: the torch.Generator, on the model's device, that draws the
    tokens, so a seeded one gives the same tokens again (None = torch's
    global one). stops: tokens after which a row's call returns. routes:
    record each row's routed experts (R3) in the model's cache (routes).
    drafter, draft, costs, outcomes: speculative decoding (module
    docstring); then rounds counts the steps that drafted, verified the
    drafts the model checked, and accepted those it kept."""

    def __init__(self, model, prompt, rows=1, *, temperature=0.8, top_k=40, generator=None, stops=(), drafter=None,
                 draft=DRAFT, costs=None, outcomes=None, routes=False):
        if not prompt:
            raise ValueError("a prompt holds at least one token")
        if costs is not None and len(costs) < draft + 1:
            raise ValueError(f"costs gives {len(costs)} pass times; drafting {draft} tokens needs {draft + 1}")
        self.model, self.temperature, self.top_k, self.generator = model, temperature, top_k, generator
        self.stops, self.drafter, self.draft, self.costs = set(stops), drafter, draft, costs
        self.outcomes = outcomes
        self.rounds = self.verified = self.accepted = 0
        self.device = next(model.parameters()).device
        self.histories = [list(prompt) for _ in range(rows)]  # every token of each row, read or not
        self.pending = [[prompt[-1]] for _ in range(rows)]  # each row's tokens the model has not read yet
        self.features = tuple(drafter.features) if drafter is not None else ()
        self.cache = self.context = None  # context: the drafter's, (rows, window, F)
        if hasattr(model, "new_cache"):
            shared = model.new_cache(1, routes=routes)
            if len(prompt) > 1:
                read = model.extend(self._tensor([prompt[:-1]]), shared, features=self.features)
                self._remember(read, [len(prompt) - 1])
            self.cache = shared.select([0] * rows)  # the prompt read once, copied into every row
            if self.context is not None:
                self.context = self.context.expand(rows, -1, -1).clone()
            elif self.features:  # a prompt of one token: nothing read yet, zeros before the text
                self.context = torch.zeros(rows, drafter.window, len(self.features) * model.config.d_model,
                                           device=self.device)
        elif drafter is not None or routes:
            raise ValueError("drafting and recording routes need a model with a generation cache")

    @property
    def routes(self):
        """Each row's recorded routes (wpt_model.Routes), or None."""
        return self.cache.routes if self.cache is not None else None

    @torch.no_grad()
    def write(self, inputs, limits):
        """inputs: {row: tokens it reads before writing (a tool result, the
        template's), possibly none}; limits: {row: the most tokens it may
        write now, at least 1}. Returns {row: the tokens it wrote}: at least
        one, at most its limit, ending at the first stop token."""
        rows = sorted(limits)
        for row in rows:
            self.pending[row] += inputs.get(row, [])
            self.histories[row] += inputs.get(row, [])
        if self.drafter is not None:
            written = self._speculate(rows, limits)
        else:
            written = {row: [token] for row, token in zip(rows, self._sample(self._scores(rows)).tolist())}
        for row, tokens in written.items():
            self.histories[row] += tokens
            self.pending[row] = [tokens[-1]]
        return written

    def _tensor(self, rows):
        width = max(len(row) for row in rows)
        return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long, device=self.device)

    def _remember(self, read, counts):
        """Keeps the drafter's context up to date once the model has read
        (read: what extend returned, with features) the first counts[b] of
        row b's new tokens: each row's last window features."""
        if not self.features:
            return
        features = read[2].float()
        if self.context is None:
            self.context = features.new_zeros(len(features), self.drafter.window, features.shape[-1])
        counts = torch.as_tensor(counts, device=self.device)
        both = torch.cat([self.context, features], 1)
        index = counts[:, None] + torch.arange(self.context.shape[1], device=self.device)
        self.context = both[torch.arange(len(both), device=self.device)[:, None], index]

    def _read(self, rows, extra=None, undoable=False):
        """The model reads each row's pending tokens (and extra[i] after
        them); returns its scores (len(rows), width, vocab), each row's at
        its own places, and what extend returned."""
        feeds = [[] for _ in self.pending]
        for i, row in enumerate(rows):
            feeds[row] = self.pending[row] + (extra[i] if extra is not None else [])
        read = self.model.extend(self._tensor(feeds), self.cache, [len(feed) for feed in feeds], undoable,
                                 features=self.features)
        return read[0][rows], read

    def _scores(self, rows):
        """The next-token scores (len(rows), vocab) of each row."""
        if self.cache is None:
            return torch.cat([self.model(self._tensor([self.histories[row]]))[:, -1] for row in rows])
        scores, _ = self._read(rows)
        ends = torch.tensor([len(self.pending[row]) - 1 for row in rows], device=self.device)
        return scores[torch.arange(len(rows), device=self.device), ends]

    def _probabilities(self, scores):
        scores = scores.float() / self.temperature
        if self.top_k is not None:
            k = min(self.top_k, scores.size(-1))
            v, _ = torch.topk(scores, k)
            scores = scores.masked_fill(scores < v[..., [-1]], float("-inf"))
        return F.softmax(scores, dim=-1)

    def _sample(self, scores):
        return torch.multinomial(self._probabilities(scores), 1, generator=self.generator)[:, 0]

    def _catch_up(self, rows):
        """The model reads every pending token of rows but the last, so that
        the drafter's context reaches each row's anchor."""
        behind = [row for row in rows if len(self.pending[row]) > 1]
        if not behind:
            return
        feeds = [self.pending[row][:-1] if row in behind else [] for row in range(len(self.pending))]
        counts = [len(feed) for feed in feeds]
        self._remember(self.model.extend(self._tensor(feeds), self.cache, counts, features=self.features), counts)
        for row in behind:
            self.pending[row] = self.pending[row][-1:]

    def _speculate(self, rows, limits):
        """One step of speculative decoding for rows (module docstring)."""
        if self.features:
            self._catch_up(rows)
        context = (self.context[rows], self.cache.lengths[rows]) if self.features else None
        drafted, q, confidences = self.drafter.draft(self.model, [self.histories[row] for row in rows], context,
                                                     self.draft, self._probabilities, self.generator)
        drafted, q = drafted.to(self.device), q.to(self.device).float()
        caps = [min(self.draft, limits[row] - 1) for row in rows]  # a row writes at most its drafts and one more
        scheduled = confidences is not None and self.costs is not None
        lengths = schedule(confidences.tolist(), caps, self.costs) if scheduled else caps
        scores, read = self._read(rows, [drafted[i, :n].tolist() for i, n in enumerate(lengths)], undoable=True)
        firsts = [len(self.pending[row]) - 1 for row in rows]  # where each row's first new token is scored
        written, kept = {}, [0] * len(self.pending)
        for i, row in enumerate(rows):
            p = self._probabilities(scores[i, firsts[i]:firsts[i] + lengths[i] + 1])
            tokens = []
            for j, token in enumerate(drafted[i, :lengths[i]].tolist()):
                u = torch.rand((), generator=self.generator, device=self.device)
                if u * q[i, j, token] < p[j, token]:  # kept with probability min(1, p / q)
                    tokens.append(token)
                    continue
                rest = (p[j] - q[i, j]).clamp(min=0)
                tokens.append(int(torch.multinomial(rest / rest.sum(), 1, generator=self.generator)))
                break
            else:  # every draft kept: one more token from the model
                tokens.append(int(torch.multinomial(p[-1], 1, generator=self.generator)))
            self.verified += lengths[i]
            self.accepted += len(tokens) - 1
            if self.outcomes is not None and confidences is not None:
                self.outcomes.append((confidences[i, :lengths[i]].tolist(), min(len(tokens) - 1, lengths[i])))
            tokens = _cut(tokens, self.stops, limits[row])
            written[row] = tokens
            kept[row] = firsts[i] + len(tokens)  # the pending tokens and every written token but the last
        self.rounds += 1
        self.model.undo(self.cache, kept)
        self._remember(read, kept)
        return written


def schedule(confidences, caps, costs):
    """DSpark's Hardware-Aware Prefix Scheduler (its Algorithm 1), for WPT's
    batch: how many drafts of each row to verify. confidences[r][j]: the
    chance that draft j of row r is kept when every draft before it was;
    caps[r]: the most of row r's drafts that may be verified; costs[n - 1]:
    the seconds of one pass that reads n tokens per row. A draft is kept
    with the product of the confidences up to it, which it adds to the
    expected tokens. WPT's pass reads every row as wide as its widest, so
    its cost follows its width alone, and a draft within the width paid for
    costs nothing: the widths are admitted from the narrowest, each with
    every row's draft at that place, while the expected tokens per second
    grow, and the first width that does not stops the admission (Algorithm
    1's early stop). Whether a draft is admitted depends only on the chances
    of the drafts up to it, never on the draft itself, which keeps the
    model's distribution unchanged."""
    survival = [1.0] * len(caps)
    expected = float(len(caps))  # each row writes at least one token
    best, width = expected / costs[0], 0
    for j in range(max(caps, default=0)):
        for r, cap in enumerate(caps):
            if j < cap:
                survival[r] *= confidences[r][j]
                expected += survival[r]
        if expected / costs[j + 1] <= best:
            break
        best, width = expected / costs[j + 1], j + 1
    return [min(width, cap) for cap in caps]


@torch.no_grad()
def measure_costs(model, rows=1, draft=DRAFT, context=256, repeats=3):
    """The seconds of one pass of model (one with a generation cache) that
    reads n tokens per row through a cache of `rows` rows that have read
    `context` tokens, for n = 1 .. draft + 1, each the fastest of `repeats`
    passes, each undone: the table a run measures once, records, and gives
    its writers as costs. The times follow this machine and its load."""
    device = next(model.parameters()).device
    cache = model.new_cache(rows)
    model.extend(torch.zeros(rows, context, dtype=torch.long, device=device), cache)
    costs = []
    for n in range(1, draft + 2):
        ids = torch.zeros(rows, n, dtype=torch.long, device=device)
        fastest = None
        for _ in range(repeats):
            _synchronize(device)
            start = time.perf_counter()
            model.extend(ids, cache, undoable=True)
            _synchronize(device)
            seconds = time.perf_counter() - start
            model.undo(cache, [0] * rows)
            fastest = seconds if fastest is None else min(fastest, seconds)
        costs.append(fastest)
    return costs


def costs_setup(model_config, rows, draft, device) -> dict:
    """What a table of pass times is measured on, and what a replay must
    match: the model's config (a dataclass, such as wpt_model.ModelConfig),
    the writer's rows, the drafts per round (the drafter's block), and the
    device, by its kind and name: the processor's for the CPU, as
    wpt_utils.device.processor_name reads it (such as "cpu (AMD Ryzen AI 7
    450 w/ Radeon 860M)"), the GPU's otherwise (such as "cuda (NVIDIA
    GeForce RTX 5080 Laptop GPU)"), never its number among several of one
    name."""
    device = torch.device(device)
    name = processor_name() if device.type == "cpu" else getattr(torch, device.type).get_device_name(device)
    return json.loads(json.dumps({"model": asdict(model_config), "rows": rows, "draft": draft,
                                  "device": f"{device.type} ({name})"}))


def costs_file(costs, setup) -> bytes:
    """The text of COSTS_FILE for a table measure_costs gave on setup
    (costs_setup)."""
    return (json.dumps({"schema": 2, "setup": setup, "seconds": costs}, indent=2) + "\n").encode()


def replayed_costs(data, setup) -> list:
    """The costs of a COSTS_FILE an earlier run recorded (its bytes), for a
    writer on setup (costs_setup); ValueError when it was measured on
    another setup, naming each difference."""
    table = json.loads(data)
    if table.get("schema") != 2:
        raise ValueError(f"the recorded costs are of shape {table.get('schema')}, which names no setup; a replay "
                         "needs shape 2")
    recorded = table["setup"]
    differences = []
    for key in setup:
        if key == "model":
            mine, theirs = setup["model"], recorded.get("model") or {}
            differences += [f"model {name} {theirs.get(name)!r}, here {mine.get(name)!r}"
                            for name in sorted(set(mine) | set(theirs)) if mine.get(name) != theirs.get(name)]
        elif recorded.get(key) != setup[key]:
            differences.append(f"{key} {recorded.get(key)!r}, here {setup[key]!r}")
    if differences:
        raise ValueError("the recorded costs were measured on another setup: " + "; ".join(differences))
    return [float(seconds) for seconds in table["seconds"]]


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cut(tokens, stops, limit):
    """tokens up to the first stop token, at most limit of them."""
    for i, token in enumerate(tokens):
        if token in stops:
            tokens = tokens[:i + 1]
            break
    return tokens[:limit]


def next_message(model, tokenizer, conversation, *, max_new_tokens, temperature=0.8, top_k=40, generator=None,
                 drafter=None, costs=None) -> list[int]:
    """The tokens the model writes for the assistant's next message after
    conversation (docs/chat-format.md), read through the chat template
    (wpt_tokenizer.chat.prompt): up to <|im_end|> or the end of a document,
    which are left out, or max_new_tokens; the sampling as sample_tokens.
    chat.reply(tokenizer, tokens) gives its blocks."""
    written = []
    for token in sample_tokens(model, chat.prompt(tokenizer, conversation), max_new_tokens=max_new_tokens,
                               temperature=temperature, top_k=top_k, generator=generator, drafter=drafter,
                               costs=costs):
        if token in (chat.IM_END, END):
            break
        written.append(token)
    return written


@torch.no_grad()
def sample_tokens(model, ids, *, max_new_tokens, temperature=0.8, top_k=40, generator=None, drafter=None, costs=None):
    """Yield up to max_new_tokens sampled token ids, one at a time, after ids
    (a list of ints, at least one), through a Writer of one row. The caller
    decides when to stop by breaking out of the iteration."""
    writer = Writer(model, ids, temperature=temperature, top_k=top_k, generator=generator, drafter=drafter,
                    draft=drafter.block if drafter is not None else DRAFT, costs=costs)  # as many as the drafter makes
    written = 0
    while written < max_new_tokens:
        for token in writer.write({}, {0: max_new_tokens - written})[0]:
            written += 1
            yield token
