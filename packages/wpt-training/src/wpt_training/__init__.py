"""Next-token training with early stopping on a smoothed validation loss."""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    steps: int = 3000         # maximum training steps
    lr: float = 3e-4
    eval_every: int = 200     # measure validation loss every N steps
    eval_batches: int = 20    # batches averaged per measurement
    patience: int = 10        # stop if smoothed val loss hasn't improved for this many evaluations
    min_improve: float = 0.01  # at Chinchilla balance, an improvement must be at least 1%
    smoothing: int = 3        # average val loss over this many evaluations to fight noise
    chinchilla: int = 20      # tokens per parameter at which training data is "enough"


def get_batch(data, block_size, batch_size, device):
    """Random windows of the data; the target is the same window shifted by one."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def loss_on(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def estimate_loss(model, data, block_size, config: TrainConfig, device):
    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        x, y = get_batch(data, block_size, config.batch_size, device)
        losses.append(loss_on(model, x, y).item())
    model.train()
    return sum(losses) / len(losses)


def train(model, train_data, val_data, *, block_size, config: TrainConfig, device, log=print):
    """Train in place, then load the best-scoring weights back into the model.

    When val_data is too short for one window, validation uses train_data.
    log receives one progress line per evaluation.
    """
    if len(val_data) <= block_size + 1:
        log("text is small: validation loss is measured on training data")
        val_data = train_data

    n_tokens = len(train_data) + len(val_data)
    n_params = sum(p.numel() for p in model.parameters())
    ratio = n_tokens / n_params
    # the smaller the tokens-per-parameter ratio, the more we tolerate overfitting:
    # at Chinchilla balance the rule is strict (1%); a data-starved run relaxes
    # toward "never stop early, memorizing is the only available mode"
    min_improve = max(0.0005, config.min_improve * min(1.0, ratio / config.chinchilla))
    log(f"tokens: {n_tokens:,} | params: {n_params:,} | ratio: {ratio:.3f} tok/param "
        f"(Chinchilla {config.chinchilla}) -> improvement threshold {min_improve*100:.2f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    best_val, best_state = float("inf"), None
    evals_without_improvement = 0
    val_history = []
    t0 = time.time()
    for step in range(1, config.steps + 1):
        x, y = get_batch(train_data, block_size, config.batch_size, device)
        loss = loss_on(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config.eval_every == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, block_size, config, device)
            val_history.append(val_loss)
            recent = val_history[-config.smoothing:]
            smoothed = sum(recent) / len(recent)
            marker = ""
            if smoothed < best_val * (1 - min_improve):
                best_val = smoothed
                evals_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = f" <- best (smoothed {smoothed:.4f})"
            else:
                evals_without_improvement += 1
            log(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | {time.time()-t0:.0f}s{marker}")
            if evals_without_improvement >= config.patience:
                log(f"early stop: smoothed val loss has not improved by {min_improve*100:.2f}% "
                    f"for {config.patience} evaluations (best {best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return best_val
