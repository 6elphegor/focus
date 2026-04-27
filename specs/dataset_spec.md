# Training Specification: Binary Increment

## Dataset

**Vocabulary:** `{0, 1, →, C, NC}`

**Task:** Given an 8-bit binary number (LSB-first, zero-padded), output its increment (+1) with explicit carry annotations, wrapping on overflow.

**Output format:** Each output position is a pair: the result bit followed by a carry token (`C` if carry-out = 1, `NC` if carry-out = 0).

Example: `111001` (LSB-first) → `0C0C0C1NC0NC1NC`

The initial carry-in is always 1. Overflow wraps modulo 2⁸.

**Fixed width:** Inputs are exactly 8 bits. Outputs are exactly 16 tokens (8 result bits each followed by one carry token).

**Splits:** 64 training examples, 192 validation examples. All 256 possible 8-bit inputs are covered. Training and validation sets are mutually exclusive. Assignment is random.

**Uniqueness:** All inputs are unique across the union of training and validation sets.

## Loss

All training examples are processed in a single batch. Loss is computed only on tokens after the `→` delimiter.

By Bayes' rule:

```
loss = (1/L) · (-ln p(data | params) - ln pdf(params))
       ╰──────── cross entropy ────────╯   ╰─ regularization ─╯
```

where `L` is the total number of tokens after the delimiter across the entire training set (here `L = 64 · 16 = 1024`). Dividing regularization by `L` means it weakens as the dataset grows — correct behavior in the Bayesian limit.

**Do not use standard weight decay from ML libraries.** The full loss above handles regularization properly.

## Metrics

All metrics are computed on tokens after the `→` delimiter.

| Metric | Definition | Computed on |
|--------|-----------|-------------|
| Cross entropy loss | `(1/L) · (-ln p(data \| params))` | Train |
| Full loss | Cross entropy + `(1/L) · (-ln pdf(params))` | Train |
| Token accuracy | Fraction of correctly predicted tokens (greedy argmax) | Train, Validation |
| Sequence accuracy | Fraction of sequences where all tokens are correct | Train, Validation |
| Effective layers | `d_eff` from the focus vector (sparsity variants only) | Train |

## Training Protocol

- Optimizer: AdamW.
- Do not train for a fixed number of iterations. Training stops when the user decides to stop (based on observing loss plateau) or the loss reaches a lower bound.
- Every `E` iterations (configurable), compute validation metrics on all 192 validation examples in a single batch.
- Display a live training graph with subplots:
  1. **Loss:** Cross entropy loss and full loss (train only).
  2. **Token accuracy:** Train and validation.
  3. **Sequence accuracy:** Train and validation.
  4. **Effective layers:** `d_eff` over training (sparsity variants only; omit for baseline).
- Support resuming training if the user decides more steps are needed after inspecting the graph.

## Saved Artifacts

| Artifact | Format | When |
|----------|--------|------|
| Model hyperparameters | JSON | Before training begins |
| Training log | CSV: `iteration, cross_entropy_loss, full_loss, train_token_accuracy, train_seq_accuracy, d_eff` | Every iteration |
| Validation log | CSV: `iteration, val_token_accuracy, val_seq_accuracy` | Every `E` iterations |
| Training graph | PNG with all subplots | On stop |
| Validation results | JSON: final token accuracy, sequence accuracy, per-example predictions | On stop |

## Why Explicit Carry Tokens?

Binary increment is inherently sequential: the output at position *k* depends on a carry chain stretching back to position 0. By interleaving carry tokens into the output, each prediction becomes a local operation — the model only needs to attend to the current input bit and the previous carry token. This converts an O(*n*)-depth serial computation into O(1)-depth per token during autoregressive generation.
