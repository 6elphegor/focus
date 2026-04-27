# Canonical Baseline vs Canonical Softmax Synthetic Experiment

## Goal

Compare two canonical hybrid-attention transformers on the `16`-bit binary increment task:

- `canonical_baseline`
- `canonical_softmax`

The two models are identical except for how they scale residual sublayers.

## Task

### Vocabulary

```text
["0", "1", "→", "C", "NC"]
```

### Example Format

Each training example starts from an integer `x` in `[0, 2^16 - 1]`.

Write `x` as a `16`-bit binary string in LSB-first order:

```text
b_0 b_1 ... b_15
```

Then append the token `"→"`, then the ripple-carry increment trace for `x + 1`:

```text
y_0 c_0 y_1 c_1 ... y_15 c_15
```

where for each bit position `i`:

- `y_i` is the output bit after increment
- `c_i` is `C` if there is a carry after bit `i`, else `NC`

So the full token sequence is:

```text
b_0 b_1 ... b_15 → y_0 c_0 y_1 c_1 ... y_15 c_15
```

Sequence lengths for `num_bits = 16`:

- prompt tokens: `16 + 1 = 17`
- supervised output tokens: `2 * 16 = 32`
- total sequence length: `49`
- autoregressive model input length: `48`

Loss is applied only on the `32` output tokens after the arrow.

## Synthetic Data

- training mode: synthetic
- batch size: `16`
- validation: disabled

At training iteration `t`, sample `16` integers uniformly from `[0, 2^16 - 1]` using a PRNG seeded with:

```text
(split_seed << 32) + t
```

For this experiment:

- `split_seed = 0`

So the synthetic data stream is deterministic across reruns.

## Shared Model Hyperparameters

- `num_bits = 16`
- `d_model = 64`
- `d_ff = 256`
- `d_attn = 64`
- `num_layers = 4`
- `attention_window_size = 8`

Important:

- here `num_layers` means canonical block pairs
- each block pair contains 4 sublayers
- total residual sublayers = `4 * num_layers = 16`

## Shared Network Structure

### Parameters

Shared by both architectures:

```text
E_tok  : [V, D]         token embeddings
W_out  : [D, V]         output projection
blocks : [L]            canonical block pairs
```

Each block pair `i` contains:

```text
Sliding attention:
  RMSNorm weight
  W_q : [D, A]
  W_k : [D, A]
  W_v : [D, D]
  tau_slide : scalar

FFN 1:
  RMSNorm weight
  W_gate_1 : [D, F]
  W_up_1   : [D, F]
  W_down_1 : [F, D]

Full attention:
  RMSNorm weight
  W_q : [D, A]
  W_k : [D, A]
  W_v : [D, D]
  tau_full : scalar

FFN 2:
  RMSNorm weight
  W_gate_2 : [D, F]
  W_up_2   : [D, F]
  W_down_2 : [F, D]
```

No bias terms are used.

### RMSNorm

For input `x ∈ R^D` and learned weight `w ∈ R^D`:

```text
RMSNorm(x; w) = (x / sqrt(mean(x^2))) * w
```

### Attention Sublayer

For normalized input `h ∈ R^(seq × D)`:

```text
Q = h W_q / sqrt(D)         # [seq, A]
K = h W_k / sqrt(D)         # [seq, A]
V = h W_v / sqrt(D)         # [seq, D]
```

Sliding attention uses RoPE on `Q` and `K`.

Full attention does not use RoPE.

Attention scores:

```text
S = tau * (Q K^T) / sqrt(A)
S = S^2
```

Masking:

- all attention is causal
- sliding attention also masks out keys more than `W = 8` positions in the past
- full attention has no sliding-window restriction

Output:

```text
Attn(h) = softmax(mask(S)) V
```

### FFN Sublayer

Let:

```text
xi = sqrt(E[silu(z)^2]) for z ~ N(0, 1) ≈ 0.596469211
```

For normalized input `h ∈ R^(seq × D)`:

```text
gate = h W_gate / sqrt(D)      # [seq, F]
up   = h W_up   / sqrt(D)      # [seq, F]
mix  = silu(gate) * up         # [seq, F]
FFN(h) = mix W_down / (xi * sqrt(F))
```

## Canonical Block-Pair Computation

For block pair `i`, define:

```text
SA_i(x) = SlidingAttention_i(RMSNorm_i_slide(x))
F1_i(x) = FFN1_i(RMSNorm_i_ffn1(x))
FA_i(x) = FullAttention_i(RMSNorm_i_full(x))
F2_i(x) = FFN2_i(RMSNorm_i_ffn2(x))
```

The only difference between the two architectures is how these four residual additions are scaled.

## Architecture 1: `canonical_baseline`

### Extra Parameters

```text
s ∈ R^(16)
```

These are learned free residual scales, one per sublayer.

Initialization:

```text
s_j = 1
```

### Forward Rule

For block pair `i`:

```text
x <- x + s_(4i+0) * SA_i(x)
x <- x + s_(4i+1) * F1_i(x)
x <- x + s_(4i+2) * FA_i(x)
x <- x + s_(4i+3) * F2_i(x)
```

After all 4 block pairs:

```text
logits = x W_out / sqrt((1 + sum_j s_j^2) * D)
```

### Regularization

```text
weight_prior_loss = 0.5 * sum(all trainable parameters squared)
d_eff = 0
```

That includes:

- all matrices
- all RMSNorm weights
- all attention temperatures
- all residual scales `s`

## Architecture 2: `canonical_softmax`

### Extra Parameters

```text
lambda ∈ R^(16)
```

These are learned sublayer focus logits, one per sublayer.

### Exact Focus Formula

First compute a softmax over all `16` canonical sublayers:

```text
p_j = exp(lambda_j) / sum_k exp(lambda_k)
```

Then l2-normalize the resulting vector:

```text
alpha_j = p_j / sqrt(sum_k p_k^2)
```

Equivalently:

```text
alpha = softmax(lambda) / ||softmax(lambda)||_2
```

These `alpha_j` are the actual residual multipliers.

### Forward Rule

For block pair `i`:

```text
x <- x + alpha_(4i+0) * SA_i(x)
x <- x + alpha_(4i+1) * F1_i(x)
x <- x + alpha_(4i+2) * FA_i(x)
x <- x + alpha_(4i+3) * F2_i(x)
```

After all 4 block pairs:

```text
logits = x W_out / sqrt((1 + sum_j alpha_j^2) * D)
```

Because `alpha` is l2-normalized:

```text
sum_j alpha_j^2 = 1
```

so in practice:

```text
logits = x W_out / sqrt(2D)
```

### Regularization

Gaussian prior term:

```text
weight_prior_loss = 0.5 * sum(all trainable parameters except lambda squared)
```

Sparsity term:

```text
d_eff = 1 / sum_j alpha_j^4
```

So `lambda` is not included in the Gaussian prior; it is regularized only through `d_eff`.

## Initialization

Use deterministic parameter initialization with `init_seed = 0`.

For every parameter except the special cases below:

```text
parameter ~ Uniform(-sqrt(3), sqrt(3))
```

Special cases:

- every RMSNorm weight vector is initialized to all ones
- every attention temperature is initialized to `1`
- in `canonical_baseline`, every residual scale `s_j` is initialized to `1`

## Training Objective

At each step, minimize:

```text
full_loss = cross_entropy_loss + (weight_prior_loss + d_eff) / train_loss_tokens
```

where:

- `cross_entropy_loss` is mean next-token cross-entropy over the supervised output tokens in the current batch
- `weight_prior_loss` and `d_eff` are architecture-dependent as defined above

For this experiment:

```text
loss_tokens_per_example = 32
regularizer_examples = 2^16 = 65536
train_loss_tokens = 65536 * 32
```

Important:

- regularizer scaling is based on full dataset size `2^16`, not synthetic batch size `16`

## Optimization

- optimizer: `AdamW`
- learning rate: `3e-4`
- `beta1 = 0.9`
- `beta2 = 0.95`
- `eps = 1e-8`
- training iterations: `5000`

## Runtime Settings

- target device: `cuda`
- `torch.compile`: enabled
- compile mode: `default`
- plot every `50` iterations
- checkpoint every `50` iterations
- training-metric logging every `50` iterations
- validation disabled

## Comparison Rule

Since validation is disabled, compare runs using final training metrics:

1. higher `train_seq_accuracy`
2. if tied, higher `train_token_accuracy`
3. if still tied, lower `full_loss`

## Functional Reproduction Command

```bash
python3 train.py \
  --architectures canonical_baseline canonical_softmax \
  --out-dir runs/compare_canonical_baseline_vs_canonical_softmax_synthetic \
  --device cuda \
  --show-plot \
  --compile-mode default \
  --validate-every 50 \
  --checkpoint-every 50 \
  --plot-every 50 \
  --train-iterations 5000 \
  --optimizer adamw \
  --learning-rate 3e-4 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --split-seed 0 \
  --init-seed 0 \
  --num-bits 16 \
  --attention-window-size 8 \
  --synthetic-train \
  --synthetic-train-batch-size 16 \
  --disable-validation \
  --d-model 64 \
  --d-ff 256 \
  --d-attn 64 \
  --num-layers 4
```
