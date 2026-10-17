# Model architecture

Maintainer's decisions, 2026-10-16. This page describes the architecture
WPT models use and why; the code is `packages/wpt-model` and the `muon`
method of `packages/wpt-training`. Every component comes from published
research (cited at the end) and is trained from scratch: no weights or
outputs of those models are used.

The design is one family for every size: a mixture-of-experts model that
reads long texts (32K tokens or more). The numbers of a given size (layers,
width, experts, table sizes) are not fixed here: each size is a model
recipe (below), since a small model may not want the ratios of a large one.

## The shape

```
token ids
  -> embedding, copied into 4 residual streams (mHC)
  -> layer 1 .. L, each:
       attention sublayer: KDA, or gated MLA every 4th layer and the last
       feed-forward sublayer: experts (hash-routed in the first 3 expert
         layers, router-chosen after), 2 shared experts
       Engram memory added before the sublayers of the layers the recipe names
       every sublayer joined to the streams by mHC
  -> streams summed, RMSNorm, output head          (next-token scores)
  -> one multi-token prediction block              (the token after next)
```

## Attention

**KDA layers** (Kimi Delta Attention; [Kimi Linear][kimi-linear], [Kimi K3][k3]). Three layers
out of four keep, per head, a fixed-size memory matrix `S` updated token by
token, so nothing grows with the text's length:

```
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
o_t = S_t^T q_t
```

- `q`, `k`, `v`: per-head projections, each through a short causal
  convolution (kernel 4) and SiLU; `q` and `k` are L2-normalized.
- `beta_t = sigmoid(W_beta x_t)`, one per head.
- Decay per channel, bounded as in K3 so tiles stay inside BF16 range:
  `g_t = -5 * sigmoid(exp(A_h) * z_t)`, `alpha_t = exp(g_t)`, where `z_t` is
  a low-rank projection of `x_t` (rank = head dimension) and `A_h` a
  learned scalar per head, starting at 0.
- Output: `W_o(sigmoid(W_g x_t) * RMSNorm_head(o_t))`.
- Head dimension 128 for keys and values in the papers; the recipe sets it.

KDA layers carry the position information; there is no positional encoding
anywhere else. Kimi Linear's ablation found 3 KDA to 1 full attention best
among 0:1, 1:1, 3:1, 7:1 and 15:1, and the hybrid beat full attention at
1.4T tokens (MMLU-Pro 51.0 against 47.2).

Training uses the chunked form (chunks of 64 tokens) from the
flash-linear-attention library on GPUs; a step-by-step version of the
formula above is the reference, used in tests, and the two must agree.
Where flash-linear-attention cannot run (it needs Triton: torch brings it on
Linux, triton-windows on Windows, and CPUs have no kernel), the same chunked
computation runs in plain PyTorch, 16 tokens at a time, factored at each chunk's middle token so no exponential leaves FP32's
range; it agrees with the reference too, and is much slower.

**Gated MLA layers** (Multi-head Latent Attention; [DeepSeek-V2][v2], [Kimi K3][k3]).
Every fourth layer, and the last, attends to the whole text. Each token's
keys and values are stored as one small compressed vector `c_t = W_c x_t`
and expanded when used, which made DeepSeek-V2's cache 93.3% smaller with
the quality of standard attention. No positional encoding (NoPE), as in
Kimi Linear and K3, so the rotary key part of DeepSeek's MLA is left out.
K3's output gate: `y_t = W_o(sigmoid(W_g x_t) * o_t)`. The attention output
is kept in FP32 during training, as K3 does.

## Experts

**Latent experts with SiTU-GLU** ([Kimi K3][k3]'s Stable LatentMoE). Routed
experts work in a narrower space:

```
z = W_down x
u = sum over chosen experts i of p_i E_i(z)
y = sum of the 2 shared experts E_shared(x) + W_up RMSNorm(u)
```

Shared experts are always active and work at full width ([DeepSeekMoE][moe]'s
idea: they hold common knowledge so routed experts can specialise). K3
uses half the width for the latent space.

Every expert is a SiTU-GLU feed-forward block, whose branches are capped by
scaled tanh functions so one unit's output never exceeds 100:

```
SiTU-GLU(x) = [4 tanh(W_g x / 4) * sigmoid(W_g x)] * [25 tanh(W_u x / 25)]
```

followed by the expert's down projection.

**Routing.** Router scores are `s = sigmoid(W_r x)`. The top `k` experts
are chosen on `s + b`; their weights are `p_i = s_i / sum of the chosen s`,
without `b`.

**Quantile Balancing** ([Kimi K3][k3]) sets the biases `b` so every expert gets
about its share of tokens, with no auxiliary loss. After each step's
forward pass, for `m` tokens, `n` experts and top `k`:

- each token's cutoff `alpha_i` is its `(k+1)`-th largest biased score;
- each expert's new bias is `-quantile_{1 - k/n}(s_j - alpha)` over the
  step's tokens, then all biases are shifted to have mean 0;
- the new biases route the next step, never the batch they came from, and
  are frozen for inference.

WPT reads the quantile from a histogram of each expert's `s_j - alpha`,
which every process adds up: 4096 bins within 0.5 of the expert's previous
quantile (`-b_j`), with values outside counted in the edge bins. A bin is
about 0.00024 wide, and one step moves a bias by at most 0.5.

**Hash routing in the first 3 expert layers** ([DeepSeek-V4][v4]). Early layers
have seen little context, so their experts are chosen from the token id
alone, by a fixed function, with equal weights `1/k`. DeepSeek-V4 does not
publish its function; WPT uses, for token id `t` and `j = 0 .. k-1`:
`expert_j(t) = (h(t) + j * n / k) mod n`, with
`h(t) = (t * 2654435761 mod 2^32) mod n`. These layers have no router and
no bias.

## Engram memory

[Engram][engram] (DeepSeek) adds a lookup that costs almost no compute: embeddings of
the last few tokens' N-grams, read from hashed tables, mixed into the
hidden state through a learned gate.

- Token ids first map to a canonical id: tokens whose decoded text is the
  same after NFKC normalization and lowercasing share one id (a map built
  from the vocabulary; tokens that are not whole UTF-8 text map to
  themselves). This ties the tables to the vocabulary.
- For each order `n = 2 .. N` (N = 3 in Engram-27B) and each of `K` hash
  heads (8 in Engram-27B), the suffix N-gram's canonical ids are hashed into
  a table of prime size and one row is read. In WPT, each (order, head)
  pair has its own table, of a distinct prime size: the smallest primes at
  or above the recipe's table size. The rows are concatenated into
  `e_t`. DeepSeek describes its hash only as multiplicative-XOR; WPT's
  constants are in the code.
- Gate: `k_t = W_K e_t`, `v_t = W_V e_t`,
  `a_t = sigmoid(RMSNorm(h_t) . RMSNorm(k_t) / sqrt(d))`, value `a_t v_t`.
  With mHC's 4 streams, `W_V` is shared and each stream has its own `W_K`.
- Then `Y = SiLU(Conv1D(RMSNorm(V))) + V`, a depthwise causal convolution
  of kernel 4 and dilation N, starting at zero so the module starts as
  nothing; `Y` is added before the layer's sublayers. WPT shares this
  convolution across the 4 streams.
- The recipe names the layers (Engram-27B: layers 2 and 15; the 3B
  reference: 2 and 6).

DeepSeek found a U-shaped law: at fixed parameters, moving 20 to 25% of the
sparse parameter budget (everything not active per token) from experts to
Engram gave the lowest loss. Lookups depend only on input tokens, so large
tables can stay in CPU memory and be fetched ahead.

## Residual streams: mHC

[mHC][mhc] (DeepSeek, used in [DeepSeek-V4][v4]) replaces the single residual sum with
`n = 4` streams. Each sublayer (attention, feed-forward) has its own
mappings, computed from the RMS-normalized, flattened streams `x'`:

```
H_pre  = sigmoid(a_pre  * x' phi_pre  + b_pre)            1 x n
H_post = 2 sigmoid(a_post * x' phi_post + b_post)         1 x n
H_res  = SinkhornKnopp(a_res * mat(x' phi_res) + b_res)   n x n
x_next = H_res x + H_post^T F(H_pre x)
```

Sinkhorn-Knopp starts from `exp` of the matrix and alternates column and
row normalization 20 times, so rows and columns sum to 1 and the product of
many layers stays stable. The gating factors `a` start at 0.01. The papers
do not say how the embedding becomes 4 streams or how they end; WPT copies
the embedding into each stream and sums the streams before the final
RMSNorm. Nor do they give the biases' starting values; WPT starts with
`H_pre` averaging the streams (`b_pre = logit(1/4)`), `H_post` adding the
output to each stream in full (`b_post = 0`) and `H_res` near the identity
(`b_res` = 4 on the diagonal, 0 elsewhere).

## Multi-token prediction

One extra block ([DeepSeek-V3][v3]'s MTP module) predicts the token after next:
it joins the final hidden state and the next token's embedding (each
RMS-normalized, concatenated, projected back to the model width), runs one
layer, and uses the shared output head. WPT's layer is a gated MLA and a
dense SiTU-GLU as wide as the two shared experts together, on one stream. Its loss is added with weight 0.3,
then 0.1 (DeepSeek-V3 switched after 10T of 14.8T tokens; the recipe sets
the switch). At generation time it can serve as a draft that guesses one
token ahead; the model works without it.

## Training

**Optimizer.** Muon for matrix weights ([Moonlight][moonlight]): the momentum is
orthogonalized by Newton-Schulz steps, then scaled by
`0.2 * sqrt(max(A, B))` for an `A x B` matrix so AdamW's learning rates
carry over, with decoupled weight decay:
`W = W - lr * (0.2 sqrt(max(A, B)) O + wd * W)`. Moonlight reports AdamW's
quality with about 52% of the compute.

- Per-head Muon ([K3][k3], [DeepSeek-V4.1][v41]): the query, key and value projections
  are orthogonalized head by head, so heads with large gradients do not
  dominate the others.
- QK-Clip ([Kimi K2][k2]): after each step, in every MLA layer, a head whose
  largest query-key score passed `tau = 100` has its query and key weights
  scaled by `sqrt(min(1, tau / max score))` each. KDA layers have no
  softmax scores and are not clipped.
- AdamW for the embedding, the output head, RMSNorm weights, and mHC's
  biases and gating factors (DeepSeek-V4's split); Adam with 5 times the
  base learning rate and no weight decay for Engram tables (DeepSeek's
  Engram setting).

**Schedule.** Multi-step ([DeepSeek LLM][llm], also used in the [mHC paper][mhc]): linear
warm-up to the maximum over 2,000 steps, then 31.6% of it after 80% of the
training tokens and 10% after 90%. DeepSeek reports final results matching
cosine decay, and a run can be extended by continuing from a checkpoint
taken before the first drop.

**Numbers.** BF16 for weights' computations on GPUs, FP32 for the weights
themselves, the optimizer state, the attention output, KDA's memory,
routing scores and mHC's mapping coefficients. CPUs compute in FP32. No
layer uses dropout, as in the papers.

**Memory.** `wpt-train --recompute` keeps only each layer's input during a
pass and runs the layer again during the backward pass: the same result
with much less memory, about a third slower (training.md). The layers'
second run records no balancing counts or query-key scores.

**Repeatability.** Runs stay the same for one seed on any hardware
(training.md): routing ties, balancing quantiles and expert gradient sums
are computed in a fixed order, over the whole global batch.

## Model recipes

A model recipe, `data/recipes/models/<name>.json`, says everything that
changes the resulting model, so a release model is rebuilt from one file
and the repository:

- the data: mixture, tokenizer recipe, total tokens and held-out share
  (what a dataset recipe held before; dataset recipes are folded into
  model recipes);
- the architecture's numbers: layers, width, context length, KDA and MLA
  heads and dimensions, expert counts and widths, top `k`, latent width,
  Engram orders, heads, table sizes and layers;
- training: steps, batch, seed, learning rate and schedule points, Muon
  and AdamW settings, the MTP loss weights;
- the post-training steps to run after pre-training.

Settings that change the result come only from the recipe. Command-line
options remain only for what does not: device strategy, number of
processes, streaming, output folder, run name, resuming, micro-batch,
recomputing, checkpoints, weight file size and restarts. Mixtures and
tokenizer recipes stay separate files, since several models share them.
A model has at least 2 layers, since KDA layers carry the position
information and the last layer is MLA.

A new recipe's name is an adjective and an animal drawn by
`uv run scripts/recipe_name.py`, among about 11.5 million names, so the
name says nothing about the GPU or size it was made for; the recipe's
`note` says that. Three recipes exist, all on the admitted public-domain
books (about 4.3M tokens) until a larger mixture exists:

- `classics-test`: 4.6M parameters, 256-token windows, a test to see every
  step work on one machine.
- `watery-tardigrade`: 101M parameters (8 layers, width 512, 32 experts
  with top 4), 32K-token windows, sized for one 16 GB GPU with
  `--recompute --micro-batch 1`. It is untested: see below.
- `unimpaired-axolotl`: half of `watery-tardigrade`, 50M parameters (the
  same 8 layers and 32 experts, width 320 with heads of 80), 32K-token
  windows, sized for about 8 GB with `--recompute --micro-batch 1`. On an integrated Radeon
  860M, one 32K window of random tokens peaked at 6.55 GiB and took 105 s,
  but a real run on the books hung that GPU in its first step, so it is
  meant for a stronger GPU; no full run yet.

## Limits of the evidence

Every result above was measured on models of 3B parameters or more, most at
27B to 2.8T, by the labs that built them; none was measured at the sizes a
single 16 GB GPU trains. A first recipe sized for such a GPU carries the
papers' ratios untested, and its first runs are the first check of whether
they hold small. The memory of `watery-tardigrade` is measured only up
to 16K tokens, on another GPU (one window: 1.5 GiB at 4K, 2.7 GiB at 8K,
4.6 GiB at 16K with `--recompute`), so 32K needs about 8 GiB by
extrapolation, not by a run; `unimpaired-axolotl` was measured at 32K
itself. The details the papers leave out (the hash routing function, Engram's hash constants and tables, how mHC's streams start and
end, the histogram behind Quantile Balancing, the MTP layer) are WPT's own
choices, marked as such above.

## Sources

- [Kimi Linear: hybrid linear attention (KDA)](https://arxiv.org/abs/2510.26692)
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [DeepSeek-V2 and MLA](https://arxiv.org/abs/2405.04434)
- [DeepSeekMoE: expert specialization](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V4: a million-token context](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4.1-Flash: KV cache compression](https://arxiv.org/abs/2609.19969)
- [Engram: conditional memory via lookup](https://arxiv.org/abs/2601.07372)
- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [Moonlight: Muon at scale](https://arxiv.org/abs/2502.16982)
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- [DeepSeek LLM: scaling laws](https://arxiv.org/abs/2401.02954)

[kimi-linear]: https://arxiv.org/abs/2510.26692
[k3]: https://arxiv.org/abs/2607.24653
[v2]: https://arxiv.org/abs/2405.04434
[moe]: https://arxiv.org/abs/2401.06066
[v3]: https://arxiv.org/abs/2412.19437
[v4]: https://arxiv.org/abs/2606.19348
[v41]: https://arxiv.org/abs/2609.19969
[engram]: https://arxiv.org/abs/2601.07372
[mhc]: https://arxiv.org/abs/2512.24880
[moonlight]: https://arxiv.org/abs/2502.16982
[k2]: https://arxiv.org/abs/2507.20534
[llm]: https://arxiv.org/abs/2401.02954
