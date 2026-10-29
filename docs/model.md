# Model architecture

This page describes the architecture WPT models use and why; the code is
`packages/wpt-model` and the `muon` optimizer of `packages/wpt-training`.
Every component comes from published research (cited at the end) and is
trained from scratch: no weights or outputs of those models are used.

The design is one family for every size: a mixture-of-experts model that
reads long texts, aiming at a context of 1M tokens, which a recipe reaches
by stages of growing windows (training.md, "Stages"). The numbers of a given size (layers, width, experts, table
sizes) are not fixed here: each size is a model recipe (below), since a
small model may not want the ratios of a large one.

**Weights whose global layers use gated MLA do not load.** Their
`config.json` names MLA numbers that `ModelConfig` refuses, and their
weights have no counterpart in CSA and HCA layers. Such a model must be
trained again from its recipe (see "Model recipes").

## The shape

```
token ids
  -> embedding, copied into 4 residual streams (mHC)
  -> layer 1 .. L, each:
       attention sublayer: KDA, or CSA or HCA every 4th layer and the
         last, the kind of each named by the recipe (global_attention)
       feed-forward sublayer: experts (hash-routed in the first 3 expert
         layers, router-chosen after), 2 shared experts
       Engram memory added before the sublayers of the layers the recipe names
       every sublayer joined to the streams by mHC
  -> streams summed, RMSNorm, output head          (next-token scores)
  -> one multi-token prediction block, window attention (the token after next)
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

KDA layers have no positional encoding: their memory is ordered by
construction. Kimi Linear's ablation found 3 KDA to 1 full attention best
among 0:1, 1:1, 3:1, 7:1 and 15:1, and the hybrid beat full attention at
1.4T tokens (MMLU-Pro 51.0 against 47.2). WPT keeps that ratio of 3 KDA
layers to 1 global layer.

Training uses the chunked form (chunks of 64 tokens) from the
flash-linear-attention library on GPUs; a step-by-step version of the
formula above is the reference, used in tests, and the two must agree.
Where flash-linear-attention cannot run (it needs Triton: torch brings it on
Linux, triton-windows on Windows, and CPUs have no kernel), the same chunked
computation runs in plain PyTorch, 16 tokens at a time, factored at each chunk's middle token so no exponential leaves FP32's
range; it agrees with the reference too, and is much slower.

**CSA and HCA layers** ([DeepSeek-V4][v4], section 2.3). Every fourth layer,
and the last, reads the whole text, but through merged entries, so its cost
grows far slower than full attention's and a 1M-token context stays
affordable. The recipe names the kind of each global layer, in order, in
its `global_attention` list: `["CSA", "HCA"]` in an 8-layer model makes
layer 4 CSA and layer 8 HCA, as V4 interleaves them. A 4-layer model has one
global layer, so one kind.

- **Merged entries.** From the layer input `h_t`, each token gives values
  `C_t = W^KV h_t` and scores `Z_t = W^Z h_t` (one value per channel), and
  every `m` tokens become one entry: per channel, the block's values
  weighted by a softmax of `Z + B` over the block, `B` a learned bias for
  each place in the block (starting at 0 in WPT; V4 does not publish its
  start). CSA merges `m = 4` tokens, and each entry also takes the block
  before it through a second pair `W^bKV, W^bZ, B^b`: 8 tokens in one
  softmax, in 1/4 as many entries. HCA merges `m' = 128` tokens without
  overlap. Each entry is RMS-normalized. Tokens after the last full block
  make no entry yet.
- **One key-value head.** Each entry serves as both key and value of every
  query head (V4's shared key-value MQA): one key-value vector per entry,
  shared by every head.
- **Queries** come from a latent, `c^Q_t = RMSNorm(W^DQ h_t)`,
  `q_t = W^UQ c^Q_t`, split into heads, and each head is RMS-normalized
  (without a learned weight, as V4's code) before the scores.
- **The indexer (CSA only)** chooses, for each query, which merged entries
  it attends to ([DeepSeek-V3.2][v32]'s lightning indexer). It merges its own
  keys `K^I` the same way, takes its queries from the same latent,
  `q^I_t = W^IUQ c^Q_t` (`n^I` heads), and head weights `w_t = W^w h_t`
  (scaled by `1/sqrt(n^I c^I)`, as V4's code), and scores every visible
  entry `I_{t,s} = sum_h w_{t,h} ReLU(q^I_{t,h} . K^I_s)`. The top `k`
  entries are kept; HCA keeps all of its entries. The indexer's queries
  and keys are rotated, then kept in FP4 (below), so `c^I` is a power of
  2. Equal scores, frequent with FP4 values and with ReLU's zeros in small
  indexers, go to the earliest entries (WPT's choice: V4 does not say),
  so the choice depends only on the visible entries, never on how many
  follow: a window in parts and the generation cache ("Generation",
  below) choose as the whole window does. One rule serves all three
  (`sparse.py`).
- **The window.** Merged entries alone would hide a query's own block, so
  each query also attends to the last 128 tokens uncompressed (itself
  included), each token's entry `RMSNorm(W^win h_j)` from a projection of
  its own (V4's code).
- **Causality.** A query at position `t` sees a merged entry only when the
  entry's last token is at or before `t` (entry `s` of ratio `m` when
  `s < (t + 1) / m`); the indexer scores only those, so neither the
  merging nor the selection lets a token see a later one.
- **Attention.** One softmax over the window and the chosen entries, scaled
  by `1/sqrt(c)`, with an attention sink: a learned logit `z'_h` per head
  whose `exp` joins the denominator, so a head can give almost no weight
  to anything (starting at 0 in WPT). The scores and the softmax run in
  FP32.
- **Position.** Rotary encoding on the last 64 dimensions of every query
  and entry (V4's partial RoPE): a token's at its position, a merged
  entry's at its block's first token. CSA and HCA (their window, merged
  entries and indexer included) use base 160,000 (V4's
  `compress_rope_theta`) stretched by YaRN with V4's values: over an
  original length of 65,536 tokens, the pairs that turn more than 32 times
  (`beta_fast`) keep their frequency, those that turn fewer than once
  (`beta_slow`) turn 16 times slower (`rope_factor`), and a linear ramp
  joins the two; with 64 rotary dimensions, pairs 0 to 15 are kept and 25
  to 31 slowed. As in V4's code, YaRN changes only the frequencies, not the
  softmax scale. The window-only block uses base 10,000 (V4's `rope_theta`)
  without YaRN, as V4 does for its window-only layers. Since entries are
  also the values, each output head is rotated back by its query's
  position, so what an entry adds depends on the distance, not the
  absolute place. KDA layers stay without position encoding. Positions
  count from the window's first token, also in a segment or slice of a
  long window (below, "Long windows").
- **Storage formats** (V4, section 2.3.4 and its inference code). Every
  key-value entry, of the window and merged, is kept in FP8 (E4M3) except
  its 64 rotary dimensions, which keep their precision: each block of 64
  values shares a scale, the power of 2 at or above the block's largest
  magnitude divided by 448. The indexer's queries and merged keys are
  rotated by a Hadamard matrix (Sylvester's, scaled by `1/sqrt(c^I)`: it
  leaves their dot products unchanged and spreads each vector over every
  dimension), then kept in FP4 (E2M1), each block of 32 values with a
  power-of-2 scale at or above its largest magnitude divided by 6, then
  held in FP8 with one such scale per vector, the form DeepSeek's FP8
  indexer kernel reads; V4 notes that this step loses nothing while the
  blocks' scales stay within a bound, here a factor of 2^13. The
  queries of the main attention are not rounded. Each value is rounded to
  its format and turned back, as V4's code simulates its quantization;
  training passes the gradient through the rounding unchanged (the
  straight-through estimator of V4's quantization-aware training). So the
  non-rotary part of `c` is a multiple of 64.
- **Grouped output.** The `n_h` head outputs form `g` groups; each group's
  `c n_h / g` values are projected to `d_g`, and the `g d_g` results to the
  model width. V4 puts no gate on the output, and WPT adds none.

The indexer's choice has no gradient, so it learns by its own loss
(DeepSeek-V3.2's sparse stage): for each query, the KL divergence from the
main attention's weights on the chosen entries (summed over heads, then
normalized over those entries) to the softmax of its scores, averaged over
queries and added over CSA layers (WPT's choice: V3.2 sums over queries,
a scale that Muon's and Adam's normalized steps mostly cancel). Its inputs
are cut off from the rest of the model, so this loss trains only the
indexer, and the language-model loss only the rest. The model computes it
only when its caller asks (`model(x, indexer_loss=True)` returns it after
the scores): every built-in training step asks, and a pass that does not
ask leaves the indexers untrained. The training loss adds the gradient of
this loss but not its value, so the logged training loss stays comparable
with the validation loss; the log shows the indexer loss beside it at each
evaluation, and `provenance.json` records its value at the step whose
weights the folder holds (`indexer_loss`). WPT is sparse from its first
step: V4 and V3.2 first train with dense attention and warm the indexer up
on all entries, a stage WPT leaves out.

V4's values, which WPT uses as they are: merging ratios 4 and 128, window
128, 64 rotary dimensions, the rotary bases and YaRN above, the FP8 and FP4
formats. The recipe sets the kind of each global layer and the numbers
that change with size (heads, head width, query latent, output groups and
their width, indexer heads and width, top `k`); V4-Flash has 64 heads of
512, a query latent of 1024, 8 groups of 1024, an indexer of 64 heads of
128 and a top 512. A head is 64 rotary dimensions and a multiple of 64
more (the FP8 blocks); an indexer head is a power of 2 above 64 (the
Hadamard rotation).

Two forms compute the attention and the indexer's scores. A PyTorch form,
which runs on every device (CPUs and any GPU, native Windows included),
gathers each query's entries a block of queries at a time, in the forward
and the backward pass, so memory holds one block's entries rather than all
of them; its backward pass is written out and tested against the formula,
and it computes the FP8 and FP4 formats in FP32 by rounding, so every
device gives the same values. On a GPU, kernels compute every pass instead,
training included, in one of two languages:

- **TileLang**, DeepSeek's own, on an NVIDIA GPU under Linux (the `cuda`
  build installs it): DeepSeek's sparse attention kernel
  (`deepseek_kernels.py`, from V4-Flash's inference code) and its FP8
  indexer kernel (`deepseek_index_kernel.py`, from V3.2's), both copied
  unchanged, and WPT's own kernels for training (`tilelang_kernels.py`),
  since DeepSeek published no backward pass: one gives each head's softmax
  denominator and each entry's weight (the indexer's target), the other the
  gradients of the queries and entries.
- **Triton**, on any other NVIDIA or AMD GPU (Linux, where PyTorch brings
  Triton, and Windows through `triton-windows` in the `cuda` build):
  translations of the same four kernels (`triton_kernels.py`).

The indexer's loss stays in PyTorch, since it reads only the chosen
entries. Which form runs is decided once, before the first use
(`wpt-train` decides it before training and logs it): neither TileLang nor
Triton installed means the PyTorch form; TileLang installed without the
CUDA compiler `nvcc` it needs (on `PATH`, or under `CUDA_HOME` or
`CUDA_PATH`), a TileLang or Triton that fails to import, or a kernel that
does not compile or run on this GPU (each is compiled and run once on a
small input) stops the run with that reason. A failure inside a later call
is an error, never a switch to the PyTorch form.

The kernels compute from BF16 queries and entries (the entries' FP8 values
are exact in BF16) and FP8 indexer queries and keys, where the PyTorch form
computes in FP32, and they add up the entries' gradients across queries in
whatever order the GPU's atomic additions take; so training and evaluations
on a machine where kernels run may give slightly different numbers from
those of a machine where they do not, and from one run to the next. Tests
check each kernel against the PyTorch form at the recipes' head counts and
top `k`, on index rows as CSA layers make them (the window, with its gaps
mid-row, then the chosen entries): the attention's output within 1% of its
largest value, its gradients within 1%, the weights and softmax
denominators within 0.01%, and the indexer's scores within 0.1%, its top
`k` as good; a whole CSA layer's output, indexer loss and gradients within
1% of the PyTorch form on the CPU. They check the PyTorch form against the
formulas of the paper.

## Long windows

Training targets 1M tokens and reaches it by stages in the recipe, with
segments on one GPU, windows split across several GPUs, and sparse
attention from the start (above). The stages are in training.md, "Stages".
A window too long for one GPU runs in parts, and both ways give exactly the
loss and gradients of the whole window at once, so they are hardware
options (training.md, "Long windows"), never part of the recipe:

- **Segments**, on one GPU (`wpt-train --segment N`): the window runs as
  consecutive segments of `N` tokens, each passing to the next what its
  layers need of the earlier tokens, and the gradients flow back across
  every boundary.
- **Slices**, across processes (`wpt-train --strategy split`):
  DeepSeek-V4's two-stage contextual parallelism ([V4][v4], section
  3.4.3). Each process holds a contiguous slice of `s` tokens; stage 1,
  process `i` sends what the next one needs of its last tokens (for CSA,
  its last block of uncompressed entries, which process `i + 1` merges
  with its own); stage 2, an all-gather of every process's merged entries
  and indexer keys gives each process the window's full set, in order,
  and the causal masks and the indexer's top `k` name the entries each
  query sees. V4 pads that set at the tail because its slices hold packed
  documents of varying lengths; WPT's slices are equal and hold whole
  blocks, so none is needed.
- **Slices in segments**, both at once (`--strategy split --segment N`):
  when a slice is longer than `N`, the window runs as consecutive parts of
  `N` tokens per process, each split across the processes; process `i`
  trains the `i`-th segment of every part, the first process takes what
  the part before handed on, and each part hands on what its last slice
  ends with, so the gradients flow back across both kinds of boundary.

What crosses a boundary is what each layer passes on to later tokens
("What a layer passes on", below), the same description the generation
cache keeps.

Across processes, KDA's memory cannot wait for the slice before.
DeepSeek-V4's two stages (above) exchange tails and merged entries, nothing
for a KDA memory; WPT combines the slices' transforms of it itself. The
memory's update is linear in the memory, `S_end = A S_start + B`, so each
process computes, in one pass over its slice, how the slice transforms a
memory (its own `A` and `B`, by starting from `[I | 0]` with values `[0 |
v]`), the processes exchange these transforms, and each composes those of
the slices before its own into its starting memory (after the memory the
part before handed on, with segments) and corrects its output. Every
process works at once, and the result is exact; it costs a pass with the
memory as wide as `dk + dv` instead of `dv`. In FP32 the composed
transforms round differently from the token-by-token memory: on the CPU,
gradients through KDA differ by up to about `6e-5` of the model's largest
gradient (the same algebra in float64 agrees to `1e-15`), and the tests of
split windows keep a tolerance of `2e-4` in FP32 (training.md, "Long
windows").

A segment or slice holds a multiple of 128 tokens (HCA's merging), so no
merged entry straddles two of them; a window's last segment may be
shorter. The losses are the parts' shares of the whole window's
(training.md, "Long windows").

### What a layer passes on

What a layer passes on to later tokens is said once, per layer, and both
training's segments and slices (`sequence.py`) and generation's cache
(`cache.py`) carry exactly that. Each module that reads earlier tokens
describes what it passes on (`passes_on()`, `carry.py`), as one of three
kinds:

- **a tail**: the last `n` time steps of a value each token has, with what
  stands before the text (zeros, or Engram's padding id);
- **entries**: merged from every `m` tokens, every one kept;
- **a memory**: a state updated token by token.

| Layer | What it passes on |
| --- | --- |
| KDA | its memory `S` per head; a tail of the last 3 inputs of each short convolution |
| CSA, HCA, the MTP block's window | a tail of the last 127 window entries; a tail of the rows not merged into an entry yet (CSA's last 8 rows, the block before the current one included; HCA's last 128); every merged entry, and in CSA every indexer key |
| Engram | a tail of the last `N - 1` canonical ids; a tail of the last `3N` inputs of its convolution, per stream |
| experts, mHC | nothing: they work token by token |

A query sees itself and the 127 entries before it in the window, so 127
is all a later token needs (DeepSeek-V4's inference code keeps 128, one
more than is read). A segment, a slice and a cache take the same values
from the same description, so they cannot drift apart: a test checks that
a segment and a cache carry exactly the described values and no other,
and that both give the whole window's results. The merging of rows into
entries is one computation for all of them: the rows passed on from
before, then this part's rows, gathered block by block. Taking tokens back
(generation, below) needs one rule per kind: a tail is cut back to the
tokens kept, a memory is replayed from the one before, and entries past a
row's tokens are never read, so they need nothing.

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
layer, and uses the shared output head. WPT's layer is attention over the
last 128 tokens alone and a dense SiTU-GLU as wide as the two shared
experts together, on one stream. Its attention is V4's (above) with no
merged entries and rotary base 10,000, as V4's own MTP block in its
published code; [MiMo-V2-Flash][mimo] chose a sliding window for its MTP
block too, so that the block stays light when it drafts tokens. It never
sees further back than 128 tokens. Its loss is added with weight 0.3,
then 0.1 (DeepSeek-V3 switched after 10T of 14.8T tokens; the recipe sets
the switch). The model works without it at generation time; speculative
decoding uses a separate drafter instead ("The drafter", below), and the
block's window is kept in the generation cache for a drafter that reads
it.

## Generation

Generation reads every token once, through a cache of what each layer
passes on ("What a layer passes on", above; `packages/wpt-model`,
`cache.py`), instead of reading the whole text again for each new token.
The cache is forward's own pass, with the tokens at each row's place: the
same code as training, which reads the earlier tokens' values from the
cache instead of from a segment before.

- **KDA** keeps its memory `S` and the last 3 inputs of each short
  convolution: a fixed size, whatever the text's length.
- **CSA, HCA and the MTP block's window** keep the window's last 127
  entries, the merged entries (FP8 but their rotary dimensions), CSA's
  merged indexer keys (FP4, held in FP8), and the rows not merged into an
  entry yet (as V4's `kv_state` and `score_state`: the current block's
  rows, and in CSA the block before it, whose second projection pair the
  next entry reads).
- **Engram** keeps each attempt's last `N - 1` canonical ids and the last
  inputs of its convolution; its tables stay on the model's device.
- The experts keep nothing, but a cache can record each attempt's routed
  experts as its tokens are read, for R3 (training.md).

Only the merged entries grow, a quarter of the tokens in CSA (and as many
indexer keys) and 1/128 in HCA, so generation goes on past `block_size`
with no limit. It is the same computation as reading the whole text, so the
same scores within rounding: on the CPU, tests check each kind of layer
(KDA, CSA, HCA, the window alone, Engram, the MTP block) and the whole
model over 300 tokens read in pieces (past `block_size`, the window and
HCA's first entries) within 0.1% of the largest score, and the tokens that
sampling draws with one seed; on a GPU, each kind of attention layer
within 3% (BF16). A rounding can still fall the other way (an FP8 or FP4
value, a near tie in the indexer's or the routers' choice), which changes
a score by more than the rounding itself; the tests allow for that.

A cache holds several rows, the attempts of one task, read together in one
batch: each row has its own length, and each call gives each row its own
number of new tokens, the rest of the row being padding that changes
nothing (KDA's padded tokens write nothing and do not decay). A prompt
shared by the attempts is read once and copied into every row: KDA's
states and V4's entries alike.

**Speculative decoding.** A drafter proposes a few tokens for each row
(`DRAFT`, 5, the most DSpark's production drafter proposes); the model
reads those worth verifying in one pass (the scheduler, "The drafter",
below); each is kept with probability `min(1, p/q)` (`p` the model's
probability, `q` the drafter's), a rejected one is replaced by a draw from
`max(0, p - q)`, normalized, and when all are kept the model adds one more
token. This leaves the model's own distribution unchanged.
The cache then undoes the rejected tokens, by kind ("What a layer passes
on"): KDA replays the kept tokens' cached inputs from the `S` before the
pass ([Kimi K3][k3]'s way; flash-linear-attention's chunked kernel on a
GPU where it imports), each tail is cut back to the kept tokens, merged
entries past a row's tokens are left unread and merged again when the row
reaches them, and the recorded routes are cut. WPT's drafter is DSpark's
("The drafter", below); the machinery takes any drafter (`wpt_inference`
says what one gives), and generation without one samples a token at a
time through the cache.

**Kernels.** Plain PyTorch on every device; on a GPU where
flash-linear-attention imports, its fused recurrent KDA kernel for fewer
than 16 new tokens and its chunked kernel for more; the attention through
the same TileLang or Triton kernels as training, where they run.

`wpt_inference.Writer` writes through the cache (`sample_tokens` for one
text); `wpt-chat`, the sample `wpt-train` prints, the drafter's training
and the attempts of the `finish-tasks` and `finish-tasks-with-less-writing`
steps use it. `wpt-chat` and `wpt-train`'s sample draft with the model's
drafter. The attempts of those steps are written by the model alone,
without a drafter: the weights change at every update, a drafter is trained
only at the end of a run, and the attempts then depend on the seed alone,
never on this machine's times.

## The drafter

The drafter is a separate model like DSpark, built with the published
values of DeepSeek's production drafter, and learned only from answers
WPT's own model writes. [DSpark][dspark] is DeepSeek's drafter for V4, on
the parallel backbone of [DFlash][dflash]. The code is `wpt_model.drafter`,
and its training `wpt_inference.drafting`; its weights live in the model
folder's `drafter/` (`config.json`, `model.safetensors`,
`provenance.json`), under the model's license and header
([licenses.md](licenses.md)).

**One unit**. The model and its drafter are one unit and a hard dependency,
with separate weights:

- `wpt-train` trains a drafter at the end of every run, after the
  recipe's last step, against the final weights, the model frozen
  (training.md, "The drafter"). The drafter is not a training step.
- The drafter's `provenance.json` names the sha256 of every weight file it
  was trained against (`target.weights`). `wpt_model.load_model` refuses a
  folder whose drafter is missing, was not saved whole, or names other
  weights (`DrafterMismatch`), and no option skips the check.
- A drafter that cannot be trained (too few prompts, a loss that is not
  finite, no draft verified) fails the run, which writes nothing.
- The prompts are the beginnings of the dataset's training documents
  (each the document start and its first `prompt_tokens` tokens; the rest
  of the document is never read), and the model alone writes the answers,
  each up to the end of a document and, when a step trained on
  conversations or tasks, also up to `<|im_end|>`, the end of a turn. A
  test checks that no text past a prompt reaches the drafter's training.

**Settings.** The model recipe's `drafter` section ("Model
recipes") gives the drafter's shape (`features`, `layers`, `block`,
`markov_rank`) and its training (`epochs`, `learning_rate`, `weight_decay`,
`warm_up`, `clip`, `anchors`, `batch`, `prompts`, `prompt_tokens`,
`answer_tokens`, `temperature`, `top_k`, `held_out`, `ece_bins`, `seed`); a
setting left out keeps its default below. The three recipes state the
training values explicitly ("Model recipes"), equal to today's defaults, so
a later change of a default does not change their drafters. `features`
numbers the model's layers from 1 to its `n_layer`;
`DrafterConfig.for_model` checks them against the model's layers when a run
starts, so `wpt-train` refuses a layer the model does not have before
any step runs.

A round drafts a block after the anchor, the last token the model wrote
and has not read yet:

- **Context** (DFlash). The hidden states of 5 target layers, spread evenly
  from the 2nd to the third-to-last (fewer in a short model: the 2nd alone
  in 4 layers), at each of the last 128 tokens the model read, side by
  side, projected to the model's width and RMS-normalized.
- **Backbone** (DSpark's production values). 3 layers, each with mHC's 4
  streams, a window attention of 128 tokens and experts. The block is the
  anchor's embedding and 4 mask vectors (DSpark's anchor-first block: 5
  places give 5 drafts), read in one pass. Every layer reads the context as
  keys and values through its own key-value projection (DFlash's KV
  injection).
- **Scores.** The target's output head, frozen, gives each place's base
  scores `U_k`; the target's embedding, frozen, reads the anchor.
- **Markov head** (DSpark's default, and its production head): `p_k(v) =
  softmax(U_k + W1[x_{k-1}] W2)`, sampled from left to right, `x_0` the
  anchor. DSpark's rank is 256 at V4's width of 4096; WPT's default rank
  follows the width, `256 * d / 4096`: 8 at width 128, 20 at 320 and 32 at
  512, as the indexer scales with the width. With a vocabulary of 8,192,
  the head holds 131K of the drafter's 1.5M parameters at width 128, 328K
  of 11.7M at 320 and 524K of 28.9M at 512; at rank 256 it held 4.2M of
  5.6M, 4.2M of 15.6M and 4.2M of 32.6M.
- **Confidence head**: `c_k = sigmoid(w . [h_k ; W1[x_{k-1}]] / T_k)`, the
  chance that draft `k` is kept when every draft before it was; `T_k` comes
  from DSpark's Sequential Temperature Scaling after training.
- **Scheduler** (DSpark's Hardware-Aware Prefix Scheduler, its Algorithm
  1). A draft is kept with the product of the confidences up to it; drafts
  are verified while the expected tokens per second of the model's pass
  grow, from a table of the pass's times, and the first that does not pay
  stops the admission, so a draft's admission never depends on the draft
  itself. The times are measured once per run of `wpt-train`
  (`wpt_inference.measure_costs`) and written into the
  run's record as `scheduler-costs.json`, with the setup they were measured
  on: the model's config, the drafter's block (the drafts per round), the
  writer's rows (1) and the device, by its kind and name (such as "cuda
  (NVIDIA GeForce RTX 5080 Laptop GPU)"), never its number among GPUs of
  one name. On the CPU, the name is the processor's, such as "cpu (AMD
  Ryzen AI 7 450 w/ Radeon 860M)", so times measured on one processor are
  never replayed on another. `wpt_utils.device.processor_name` reads it
  from what each system reports: the first `model name` line of
  `/proc/cpuinfo` on Linux, the registry's `ProcessorNameString` of the
  first processor on Windows, `sysctl -n machdep.cpu.brand_string` on
  macOS, with runs of spaces made one (`platform.processor()` is empty on
  many Linux systems and names only the architecture elsewhere). A Linux
  `/proc/cpuinfo` without a `model name` line, as on most Arm processors,
  gets a name built from its `CPU implementer` and `CPU part` codes, as
  `lscpu` does: the implementer, then each kind of core once, such as "cpu
  (Arm Limited Cortex-X925 + Cortex-A725)". WPT writes its own table of
  these names from the facts Arm publishes (the implementers in Arm's
  register reference, DDI 0601, `MIDR_EL1`; each part in its core's
  Technical Reference Manual), never from `lscpu`'s code, and keeps it to
  the cores of common Arm machines that run PyTorch under Linux; a part it
  does not hold is named by its implementer and its code in hexadecimal,
  such as "Arm Limited part 0xd0d", and an implementer Arm does not publish
  by its code, such as "implementer 0x61 part 0x22". Only when those codes
  are missing too is it the machine's architecture, such as `aarch64`, the
  last resort. On macOS, a `sysctl` that is not on the search path never
  stops a run: the name is then the architecture. A file recorded on
  the CPU before, whose device is `cpu` alone, is refused like any other
  setup. `wpt-train
  --scheduler-costs FOLDER` replays the times an earlier run recorded in its record folder `FOLDER`
  instead of measuring them, and `wpt-chat --scheduler-costs FOLDER` reads
  them instead of measuring them when it loads the model; each refuses
  times recorded on another setup, naming each difference, `wpt-train`
  before any step runs. A writer given no
  times verifies every draft.

  What a replay guarantees, exactly: the scheduler uses the recorded
  seconds, so for the same drafter confidences it admits the same drafts
  as the run that measured them. With the same weights (model and
  drafter), the same prompt, the same sampling settings and the same
  seed, on a device whose passes give the same numbers, the writer then
  verifies the same drafts and draws the same tokens as a writer given
  that same table. The setup check compares only what the file records:
  the model's config, the block, the rows and the device's kind and name.
  It does not compare the weights, the drivers, the library versions or
  the machine's load, and it does not check that the times are true for
  this machine: times that are wrong here make writing slower, never
  change the model's distribution.

Training (DSpark, section 3.3): the model writes an answer to each prompt,
then reads each answer whole; random anchors in the answer start blocks
of 5; the drafter learns
`0.1 L_ce + 0.9 L_tv + 1.0 L_conf`, each place `k` (from 1) weighed
`exp(-(k - 1) / 5)`: cross-entropy on the answer's tokens, the total
variation distance to the model's distribution, and binary cross-entropy
of the confidence against `c*_k = 1 - ||p_d - p_t||_1 / 2`. The target is
frozen; only the drafter learns.

Details DSpark leaves open, and WPT's choice for each:

- Width: the model's, which sharing its embedding and head requires.
- The backbone's layers: V4's attention with no merged entries, as the
  multi-token prediction block's (rotary base 10,000), and the model's
  experts (latent SiTU-GLU experts, 2 shared, the model's numbers), every
  layer routed by its router and balanced by Quantile Balancing (no hash
  routing); DSpark says only "MoE layers with mHC".
- Each target layer's hidden state: its 4 output streams summed, as the
  model's last layer is read.
- What each place sees: the whole block both ways (DFlash), and the context
  tokens within 128 positions of it, so the window is each place's own.
- The mask vector: learned, starting from a normal draw at the embedding's
  scale. The final RMSNorm before the head: the drafter's own.
- Starting values: `W2` at zero, so the Markov head starts as nothing; the
  confidence projection has a bias.
- The drafts' distribution: the writer's temperature and top `k` applied to
  the drafter's scores, so it follows what the model samples.
- Calibration: on real outcomes. On the held-out prompts, the drafter
  drafts for the model, every draft verified, its temperatures at 1, and
  each round records each draft's confidence and whether the model kept it
  (a draft after a rejected one counts as not kept). Each position's
  temperature is chosen among 200 values evenly spaced in log from 0.05 to
  20 to lower the expected calibration error over 15 equal-width bins on
  those outcomes, position by position (Sequential Temperature Scaling).
  The same rounds measure how many drafts the model keeps, recorded in the
  drafter's provenance.
- The scheduler: DSpark's production server schedules from the confidences
  of two steps before and drops the early stop, for its zero-overhead
  scheduling, which WPT does not have; WPT runs the lossless Algorithm 1 as
  published. WPT's batch reads every row as wide as its widest, so a pass's
  cost follows its width alone: widths are admitted from the narrowest,
  each with every row's draft at that place, instead of drafts one by one
  (a draft inside a width already paid for costs nothing). The time table
  holds, for each width, the fastest of 3 passes through a cache that has
  read 256 tokens, measured once per run for one row.
- A row whose pending tokens are more than its anchor (a tool result) first
  has them read in a pass of their own, so the drafter's context reaches
  its anchor.
- Training values DSpark does not give come from DFlash: AdamW at 6e-4,
  gradients clipped at 1, a cosine schedule after a 4% warm-up, 512 anchors
  per answer redrawn each epoch; DSpark's 10 epochs; weight decay 0, 8
  answers per update, 256 prompts of at most 64 tokens each drawn from the
  training documents by a seeded reservoir, answers of at most 256 tokens
  written at temperature 1 without top `k`, and 10% of the prompts held out
  (WPT's).

Conflicts with DeepSeek-V4's published values: none; the drafter replaces
V4's MTP-1 for drafting, as DSpark replaced it in V4's production. A cost
of the scheduler: it follows this machine's measured times, so a seeded
generator may draw other tokens from one run to the next (never another
distribution), unless the run replays the times an earlier one recorded.
The attempts of the task steps are written without a drafter, so they
never depend on the times.

## Parts

A training step locks parts of the model by name ([training.md](training.md),
"Locked parts"). The model gives its parts these names (`wpt_model.PARTS`),
and every parameter belongs to exactly one of them (`WPT.parts`):

- `embedding`: the token embedding;
- `attention`: every layer's attention (KDA, CSA or HCA, with its norm),
  its indexer aside;
- `indexers`: the CSA indexers;
- `routers`: the routers of the experts layers;
- `balancing`: the routers' balancing biases, which no gradient trains:
  Quantile Balancing sets them after each update, unless a step locks
  them;
- `experts`: the experts, shared and routed, with their norm;
- `engram`: the Engram memories;
- `streams`: mHC's mappings of the residual streams;
- `final-norm`: the last norm;
- `output-head`: the output head;
- `multi-token-prediction`: the MTP block.

## Training

**Optimizer.** Muon for matrix weights ([Moonlight][moonlight]), as
[DeepSeek-V4][v4] runs it (its Algorithm 1): the momentum (with Nesterov's
trick) is divided by its Frobenius norm and orthogonalized by hybrid
Newton-Schulz steps, 8 with coefficients `(3.4445, -4.7750, 2.0315)` that
bring the singular values near 1, then 2 with `(2, -1.5, 0.5)` that settle
them at 1; the result is scaled by `0.18 * sqrt(max(A, B))` for an
`A x B` matrix, so the update's RMS is 0.18 and AdamW's learning rates
carry over, with decoupled weight decay:
`W = W - lr * (0.18 sqrt(max(A, B)) O + wd * W)`. Moonlight reports
AdamW's quality with about 52% of the compute.

- Per-head Muon ([K3][k3], [DeepSeek-V4.1][v41]): KDA's query, key and value
  projections, and the query projections of CSA, HCA and the indexer, are
  orthogonalized head by head, so heads with large gradients do not
  dominate the others. CSA and HCA have one key-value head, so their
  key-value projection is one matrix.
- Each logically separate matrix apart, as DeepSeek-V4's Muon takes "each
  logically independent weight": each output group's projection, and a
  merging's two projection pairs (`W^aKV` and `W^bKV`, `W^aZ` and `W^bZ`).
- No QK-Clip: DeepSeek-V4 leaves it out, since CSA and HCA RMS-normalize
  their queries and entries, which keeps the scores from exploding. KDA
  has no softmax scores.
- AdamW for the embedding, the output head, RMSNorm weights, and mHC's
  biases and gating factors (DeepSeek-V4's split), and for the attention
  sinks and the merging's positional biases (biases and scaling factors
  take AdamW in [DeepSeek-V4.1][v41]); Adam with 5 times the base learning
  rate and no weight decay for Engram tables (DeepSeek's Engram setting).

**Schedule.** Multi-step ([DeepSeek LLM][llm], also used in the [mHC paper][mhc]): linear
warm-up to the maximum over 2,000 updates, then 31.6% of it after 80% of the
training tokens and 10% after 90%. DeepSeek reports final results matching
cosine decay, and a run can be extended by continuing from a checkpoint
taken before the first drop.

**Numbers.** BF16 for weights' computations on GPUs, FP32 for the weights
themselves, the optimizer state, the attention scores and output, the
merging of entries, KDA's memory, routing scores and mHC's mapping
coefficients; the attention's entries in the FP8 and FP4 formats above,
simulated in FP32. The kernels, where they run, read queries and entries
in BF16 and the indexer's in FP8. CPUs compute in FP32. No layer uses dropout, as in the
papers.

**Memory.** `wpt-train --recompute` keeps only each layer's input during a
pass and runs the layer again during the backward pass: the same result
with much less memory, about a third slower (training.md). The layers'
second run records no balancing counts, and the indexer loss counts once.

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
- the architecture's numbers: layers, width, KDA heads and
  dimensions, the kind of each global layer (`global_attention`), the
  attention heads and dimensions of CSA and HCA (query
  latent, output groups and their width, indexer heads and dimensions, the
  top `k` entries), expert counts and widths, top `k`, latent width,
  Engram orders, heads, table sizes and layers;
- the `seed`;
- the `steps`, in order, each a step with, when they differ from the step
  file's, its optimizer and that optimizer's settings, its settings and
  the parts it locks ([training.md](training.md), "Steps"): the
  `read-text` step's stages (each its window length, updates and batch, in
  order), evaluations, MTP loss weights and Muon settings among them; the
  model's context is the last window length of the last step that reads
  text windows;
- the drafter (`drafter`): its shape and how it is trained ("The
  drafter"; an empty section keeps every default).

Settings that change the result come only from the recipe. Command-line
options remain only for what does not: device strategy (splitting windows
across processes included), number of processes, streaming, output folder,
run name, resuming, micro-batch, recomputing, segments, checkpoints, weight
file size and restarts. Mixtures and tokenizer recipes stay separate files,
since several models share them. A model has at least 2 layers: the last is
CSA or HCA, the others start with KDA. `wpt-dataset migrate` brings a recipe of an earlier
shape to the current one, step by step. A recipe with the MLA numbers gets
the CSA and HCA ones at 0 and an empty `global_attention` (gated MLA's
numbers say neither), which a person then fills in; `wpt-dataset check`
names each file until then. A recipe with the model's `block_size` and the
training's `steps` and `batch_size` gets one stage of the training's
`stages` (training.md, "Stages"). A recipe without the `drafter` section
gets it, after `training`, with the drafter's training values every recipe
states: 256 prompts of at most 64 tokens, answers of at most 256 tokens, 10
epochs, learning rate 6e-4, 10% held out, 512 anchors, 8 answers per update,
seed 0. The three recipes hold exactly these values; the settings a section
leaves out (the drafter's shape, `weight_decay`, `warm_up`, `clip`,
`temperature`, `top_k`, `ece_bins`) keep their defaults. A recipe with
`training` and `posttrain` gets one list, `steps`, with `seed` beside it: the
training becomes the first step, `read-text`, its method the step's
optimizer (with its settings but the MTP loss weights, which become the
step's), its stages and evaluations the step's settings, nothing locked;
each post-training step follows under the name of the step that does the
same (`sft` as `answer-as-assistant`, `grpo` as `finish-tasks`,
`token-efficiency` as `finish-tasks-with-less-writing`), its learning
rate and weight decay given to its optimizer, `adamw`, and a `tag` setting
naming one of those steps renamed with it. A step given as a Python file
keeps its name and settings, and the check that follows says what the
file lacks.

A new recipe's name is an adjective and an animal drawn by `uv run
scripts/recipe_name.py`, among about 11.5 million names, so the name says
nothing about the GPU or size it was made for; the recipe's `note` says
that. Four recipes exist, each of one stage (none yet sets V4's 4K, 16K,
64K and 1M stages). The first three read the admitted public-domain books
(about 4.3M tokens); `interstitial-pecker` reads every admitted document:

- `classics-test`: 4.8M parameters, 256-token windows, a test to see every
  step work on one machine. Its one global layer is CSA (4 heads of 128, an
  indexer of 2 heads of 128), with a top 16 of its 64 entries so the choice
  is exercised even in short windows. With 4 layers it has only one global
  layer, so it never runs HCA.
- `watery-tardigrade`: 102M parameters (8 layers, width 512, 32 experts
  with top 4; layer 4 CSA and layer 8 HCA with 8 heads of 128, a query
  latent of 128, 2 output groups of 256, an indexer of 8 heads of 128 and
  top 64), 32K-token windows, sized for one 16 GB GPU with `--recompute
  --micro-batch 1`. One 32K window of random tokens peaked at 8.24 GiB on
  an integrated Radeon 860M (training.md). It is untested in a full run:
  see below.
- `unimpaired-axolotl`: half of `watery-tardigrade`, 50M parameters (the
  same 8 layers and 32 experts, width 320, KDA heads of 80; layer 4 CSA and
  layer 8 HCA with 4 heads of 128, a query latent of 80, 1 output group of
  320, an indexer of 5 heads of 128 and top 40), 32K-token windows, sized
  for about 8 GB with `--recompute --micro-batch 1`. On an integrated
  Radeon 860M, one 32K window of random tokens peaked at 6.57 GiB
  (training.md); it is meant for a stronger GPU. No full run yet.
- `interstitial-pecker`: 146M parameters, the smallest of a series for
  watching behaviors appear as size grows, not for release. It keeps
  `watery-tardigrade`'s width of 512 and its experts and has 10 layers
  (layers 4 and 10 CSA, layer 8 HCA), since depth matters more than width
  below 1B parameters ([MobileLLM](https://arxiv.org/abs/2402.14905)). Its
  vocabulary is the `admitted` mixture's 32,768 tokens, whose two tables
  take 33.6M parameters. It reads 20 tokens per parameter (about 2.93B) as
  2,790 updates of 32 windows of 32K tokens. Both heads' scores over one
  32K window take about 8 GiB in 32-bit, so one 16 GB GPU is estimated to
  need `--recompute --micro-batch 1 --segment 8192`; nothing is measured
  yet. No full run yet.

V4's head width of 512 and its ratios (a latent of a quarter of the width,
8 heads per output group) do not fit widths of 128 to 512, so the heads are
128 wide (64 of them rotary), the query latent a quarter of the width, and
the output groups together as wide as the model. The indexer follows
V4-Flash's in proportion to the width (4096 there): `64 * d / 4096` heads
and a top `512 * d / 4096` entries, each head 128 wide as V4's; so 2 heads
and top 16 at width 128, 5 and 40 at 320, 8 and 64 at 512.

## Limits of the evidence

Every result above was measured on models of 3B parameters or more, most at
27B to 2.8T, by the labs that built them; none was measured at the sizes a
single 16 GB GPU trains. A first recipe sized for such a GPU carries the
papers' ratios untested, and its first runs are the first check of whether
they hold small. CSA and HCA in particular were built for a million
tokens and trained by DeepSeek with a dense stage first; WPT trains them
sparse from the first step, at 32K tokens and fewer, which no paper has
measured. No WPT model has yet trained beyond 32K tokens: the stages to
1M, the segments and the slices are measured only in tests and in single
steps on random tokens (training.md, "Long windows"). The drafter's
production values were set for V4-Flash and V4-Pro; WPT scales its Markov
head with the width ("The drafter"), and no drafter has yet been measured
on a real WPT run. YaRN's values and the indexer's FP4 were set by DeepSeek for
its sizes and contexts; the indexer scaled down in proportion to the width
is WPT's choice. The memory of both 32K recipes is measured on one window
of random tokens (training.md), not in a real run. The details
the papers leave out (the hash routing function, Engram's hash constants
and tables, how mHC's streams start and end, the histogram behind Quantile
Balancing, the MTP layer, the starting values of the attention sinks and
merging biases, how the indexer's loss is scaled) are WPT's own choices,
marked as such above.

## Sources

- [Kimi Linear: hybrid linear attention (KDA)](https://arxiv.org/abs/2510.26692)
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [DeepSeekMoE: expert specialization](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V4: a million-token context](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4.1-Flash: KV cache compression](https://arxiv.org/abs/2609.19969)
- [DeepSeek-V3.2: DeepSeek Sparse Attention and its lightning indexer](https://arxiv.org/abs/2512.02556)
- [MiMo-V2-Flash: a sliding-window MTP block](https://arxiv.org/abs/2601.02780)
- [Engram: conditional memory via lookup](https://arxiv.org/abs/2601.07372)
- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [Moonlight: Muon at scale](https://arxiv.org/abs/2502.16982)
- [DeepSeek LLM: scaling laws](https://arxiv.org/abs/2401.02954)
- [DSpark: confidence-scheduled speculative decoding](https://arxiv.org/abs/2607.05147)
- [DFlash: block diffusion for flash speculative decoding](https://arxiv.org/abs/2602.06036)

[kimi-linear]: https://arxiv.org/abs/2510.26692
[k3]: https://arxiv.org/abs/2607.24653
[moe]: https://arxiv.org/abs/2401.06066
[v3]: https://arxiv.org/abs/2412.19437
[v4]: https://arxiv.org/abs/2606.19348
[v41]: https://arxiv.org/abs/2609.19969
[v32]: https://arxiv.org/abs/2512.02556
[mimo]: https://arxiv.org/abs/2601.02780
[engram]: https://arxiv.org/abs/2601.07372
[mhc]: https://arxiv.org/abs/2512.24880
[moonlight]: https://arxiv.org/abs/2502.16982
[llm]: https://arxiv.org/abs/2401.02954
[dspark]: https://arxiv.org/abs/2607.05147
[dflash]: https://arxiv.org/abs/2602.06036
