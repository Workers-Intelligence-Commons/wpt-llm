"""Training a model's drafter (docs/model.md, "The drafter";
wpt_model.Drafter). A model and its drafter are one unit: wpt-train trains
one at the end of every run, against the final weights, and wpt-posttrain
a fresh one at the end of every run, against the post-trained weights.
The model is frozen while it does: only the drafter learns.

The prompts are the beginnings of the dataset's training documents: of
every document read once, `prompts` drawn at random (a seeded reservoir),
each the document's first `prompt_tokens` tokens after the document start
(the END token, as wpt-chat begins a text). The model alone writes each
answer, sampling at `temperature` (and `top_k`) with no drafter, up to a
stop token or `answer_tokens`: the rest of each document, the text people
wrote, is never kept, and the drafter learns to propose what the model
writes (DSpark's and DFlash's regenerated responses). A share of the
prompts (`held_out`, at least one) is kept apart.

Each answer is read whole by the model, which gives its hidden states at
the drafter's feature layers and its distribution of each next token. From
random anchors in the answer (`anchors` at most per answer, drawn again
each epoch, as DFlash), the drafter learns DSpark's losses: cross-entropy on
the answer's tokens, the total variation distance to the model's
distribution, and the confidence head's prediction of the chance each
draft is kept (Drafter.losses). AdamW, gradients clipped, a cosine schedule
after a warm-up (DFlash's training values). The experts' routing biases
follow Quantile Balancing after each update, as the model's.

On the held-out prompts, the drafter then drafts for the model, every draft
verified, and each round records each draft's confidence and whether the
model kept it (a draft after a rejected one counts as not kept): Sequential
Temperature Scaling sets each position's temperature on those real
outcomes (drafter.calibrate), and the same rounds measure how many drafts
the model keeps.

The settings come from the model recipe's `drafter` section (settings);
what it leaves out keeps the defaults below. Every failure (too few
prompts, a loss that is not finite) raises DrafterFailed: the run that
asked fails and writes nothing.
"""

import math
import random
from dataclasses import asdict, dataclass, fields

import torch

from wpt_model.drafter import Drafter, DrafterConfig, calibrate, temperature_grid, training_blocks

from . import Writer


class DrafterFailed(Exception):
    """The drafter could not be trained; the reason says why."""


@dataclass(frozen=True)
class Settings:
    """How a drafter is trained: the model recipe's `drafter` section, but
    its shape (DrafterConfig: features, layers, block, markov_rank)."""
    epochs: int = 10            # passes over the answers (DSpark trains each drafter for 10 epochs)
    learning_rate: float = 6e-4  # AdamW, DFlash's
    weight_decay: float = 0.0
    warm_up: float = 0.04       # share of the updates the learning rate rises over, then a cosine decay to 0 (DFlash)
    clip: float = 1.0           # gradient norm clipping (DFlash)
    anchors: int = 512          # anchors drawn from each answer per epoch, at most one per token (DFlash)
    batch: int = 8              # answers per update
    prompts: int = 256          # training documents whose beginnings are prompts
    prompt_tokens: int = 64     # the most tokens of a document's beginning in its prompt
    answer_tokens: int = 256    # the most tokens of each answer the model writes
    temperature: float = 1.0    # the answers come from the model's own probabilities, which the drafter learns
    top_k: int | None = None
    held_out: float = 0.1       # share of the prompts kept apart for calibration and the measure, at least one
    ece_bins: int = 15          # equal-width bins of the expected calibration error
    seed: int = 0               # the drafter's starting weights, the prompts and their split, the anchors, the answers


SHAPE = ("features", "layers", "block", "markov_rank")  # the recipe's drafter settings that are DrafterConfig's


def settings(model_config, section) -> tuple[DrafterConfig, Settings]:
    """(the drafter's shape, how it is trained) from a model recipe's
    `drafter` section; ValueError for a setting neither knows."""
    known = SHAPE + tuple(field.name for field in fields(Settings))
    unknown = sorted(set(section) - set(known))
    if unknown:
        raise ValueError(f"drafter settings: {', '.join(unknown)} unknown; known: {', '.join(known)}")
    shape = {name: section[name] for name in SHAPE if name in section}
    return (DrafterConfig.for_model(model_config, **shape),
            Settings(**{name: value for name, value in section.items() if name not in SHAPE}))


def beginnings(documents, count, length, end, seed):
    """The prompts: of documents (token sequences, each ending with end),
    count drawn at random with a seeded reservoir, each [end] then its
    first `length` tokens before its own end; documents with no token
    before their end are passed over. Only those first tokens are kept."""
    chosen, seen, draw = [], 0, random.Random(seed)
    for tokens in documents:
        text = min(length, len(tokens) - 1)
        if text < 1:
            continue
        prompt = [end, *(int(token) for token in tokens[:text])]
        seen += 1
        if len(chosen) < count:
            chosen.append(prompt)
        else:
            place = draw.randrange(seen)
            if place < count:
                chosen[place] = prompt
    return chosen


def train_drafter(model, documents, config: DrafterConfig, settings: Settings, *, end, stops, device, telemetry):
    """A drafter of that shape trained for model on the beginnings of
    documents (module docstring), in eval mode, with its provenance (what
    made it; the caller adds the weights it was trained against). end: the
    document-start token; stops: the tokens that end an answer. The model
    is left as it was, frozen meanwhile. Raises DrafterFailed."""
    prompts = beginnings(documents, settings.prompts, settings.prompt_tokens, end, settings.seed)
    if len(prompts) < 2:
        raise DrafterFailed(f"{len(prompts)} prompt: the drafter needs at least 2 documents with text, to learn from "
                            "one and calibrate on another")
    random.Random(settings.seed).shuffle(prompts)
    held = max(1, round(settings.held_out * len(prompts)))
    if held >= len(prompts):
        raise DrafterFailed(f"held_out {settings.held_out} keeps every one of the {len(prompts)} prompts apart")
    training_mode = model.training
    frozen = [p for p in model.parameters() if p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    model.eval()
    try:
        return _train(model, prompts, held, config, settings, stops, device, telemetry)
    finally:
        for p in frozen:
            p.requires_grad_(True)
        model.train(training_mode)


def _train(model, prompts, held, config, settings, stops, device, telemetry):
    sampling = torch.Generator(device=device).manual_seed(settings.seed)
    with telemetry.span("drafter.answers", prompts=len(prompts)):
        answers = [_answer(model, prompt, settings, stops, sampling) for prompt in prompts]
    training, held_out = answers[held:], answers[:held]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings.seed)
        drafter = Drafter(model.config, config).to(device)
    shape = {**asdict(config), "features": list(config.features)}
    updates = settings.epochs * math.ceil(len(training) / settings.batch)
    telemetry.event("drafter.settings", settings=asdict(settings), drafter=shape, prompts=len(prompts),
                    training=len(training), held=len(held_out), tokens=sum(len(ids) - start for ids, start in answers))
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    warm = max(1, math.ceil(settings.warm_up * updates))
    rate = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: (u + 1) / warm if u < warm else
                                             0.5 * (1 + math.cos(math.pi * (u - warm) / max(1, updates - warm))))
    anchors = torch.Generator().manual_seed(settings.seed)
    order = random.Random(settings.seed)
    losses = []
    drafter.train()
    for epoch in range(1, settings.epochs + 1):
        chosen = list(range(len(training)))
        order.shuffle(chosen)
        totals = []
        for first in range(0, len(chosen), settings.batch):
            blocks = [_blocks(model, drafter, *training[i], settings.anchors, anchors, device)
                      for i in chosen[first:first + settings.batch]]
            loss, parts = drafter.losses(model, *[torch.cat(part) for part in zip(*blocks)])
            if not math.isfinite(loss.item()):
                raise DrafterFailed(f"epoch {epoch}: the loss is {loss.item()}; lower the drafter's learning_rate")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), settings.clip)
            optimizer.step()
            rate.step()
            drafter.after_update(lambda tensor, op: tensor)  # one process
            totals.append(loss.item())
            telemetry.event("drafter.update", values={"loss": loss.item(), **{k: v.item() for k, v in parts.items()}},
                            epoch=epoch, update=len(totals), answers=len(blocks))
        losses.append(sum(totals) / len(totals))
        telemetry.log(f"drafter epoch {epoch}/{settings.epochs}: mean loss {losses[-1]:.4f} over {len(training)} "
                      "answers")
    drafter.eval()

    with telemetry.span("drafter.calibration"):
        outcomes, measure = _outcomes(model, drafter, [ids[:start] for ids, start in held_out], settings, stops,
                                      sampling)
        logits, kept, valid = _table(outcomes, drafter.block, device)
        if not len(logits):
            raise DrafterFailed(f"no draft was verified on the {len(held_out)} held-out prompts: nothing to "
                                "calibrate on (raise answer_tokens or prompts)")
        temperatures, before, after = calibrate(logits, kept, valid, settings.ece_bins, temperature_grid())
        drafter.temperatures.copy_(temperatures)
        telemetry.event("drafter.calibration", temperatures=temperatures.tolist(), ece_before=before, ece_after=after,
                        rounds=len(logits))
    telemetry.event("drafter.acceptance", **measure)
    telemetry.log(f"drafter: {measure['accepted']} of {measure['verified']} drafts kept on {len(held_out)} held-out "
                  f"prompts, {measure['tokens_per_round']:.2f} tokens per round of the model")
    drafter.provenance = {
        "schema": 2,
        "drafter": shape,
        "settings": asdict(settings),
        "prompts": {"count": len(prompts), "training": len(training), "held_out": len(held_out),
                    "answer_tokens": sum(len(ids) - start for ids, start in answers)},
        "losses": losses,  # each epoch's mean
        "calibration": {"temperatures": temperatures.tolist(), "ece_before": before, "ece_after": after,
                        "rounds": len(logits)},
        "acceptance": measure,
    }
    return drafter


def _answer(model, prompt, settings, stops, generator):
    """(the prompt and the model's answer to it, where the answer starts):
    the model alone writes it, until a stop token or answer_tokens."""
    writer = Writer(model, prompt, temperature=settings.temperature, top_k=settings.top_k, generator=generator,
                    stops=stops)
    answer = []
    while len(answer) < settings.answer_tokens and not (answer and answer[-1] in stops):
        answer += writer.write({}, {0: settings.answer_tokens - len(answer)})[0]
    return prompt + answer, len(prompt)


@torch.no_grad()
def _model_reads(model, drafter, ids, device):
    scores, features = model(torch.tensor([ids], device=device), features=drafter.features)
    return scores[0], features[0]


def _blocks(model, drafter, ids, start, most, generator, device):
    """The blocks of one answer for Drafter.losses: anchors drawn among the
    tokens whose next token is the answer's, at most `most`."""
    scores, features = _model_reads(model, drafter, ids, device)
    places = len(ids) - start  # anchors start - 1 .. len(ids) - 2
    chosen = torch.randperm(places, generator=generator)[:most] + start - 1
    return training_blocks(features, torch.tensor(ids, device=device), scores, chosen.to(device), drafter.block)


def _outcomes(model, drafter, prompts, settings, stops, generator):
    """The drafter drafting for the model on prompts, every draft verified,
    its temperatures at 1: each round's (confidences, drafts kept), and
    the drafts verified and kept, the rounds, and tokens written per round."""
    drafter.temperatures.fill_(1.0)  # the confidences are then the head's own, which calibrate scales
    outcomes, verified, accepted, rounds = [], 0, 0, 0
    for prompt in prompts:
        writer = Writer(model, prompt, temperature=settings.temperature, top_k=settings.top_k, generator=generator,
                        stops=stops, drafter=drafter, draft=drafter.block, outcomes=outcomes)
        written = []
        while len(written) < settings.answer_tokens and not (written and written[-1] in stops):
            written += writer.write({}, {0: settings.answer_tokens - len(written)})[0]
        verified, accepted, rounds = verified + writer.verified, accepted + writer.accepted, rounds + writer.rounds
    return outcomes, {"verified": verified, "accepted": accepted, "rounds": rounds,
                      "tokens_per_round": (accepted + rounds) / max(1, rounds)}


def _table(outcomes, block, device):
    """(logits, kept, valid) (N, block) of the rounds that verified a draft."""
    rows = [(confidences, kept) for confidences, kept in outcomes if confidences]
    logits = torch.zeros(len(rows), block, device=device)
    kept = torch.zeros_like(logits)
    valid = torch.zeros_like(logits)
    for i, (confidences, count) in enumerate(rows):
        n = len(confidences)
        logits[i, :n] = torch.logit(torch.tensor(confidences, dtype=torch.float64), eps=1e-7).float()
        kept[i, :count] = 1
        valid[i, :n] = 1
    return logits, kept, valid
