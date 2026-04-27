# Ablation Results (2026-03-22)

All runs: 8-bit binary increment, 64 fixed training examples, AdamW lr=0.03, cosine schedule, 5k iterations.

## Width Ablation (4 layers, baseline)

| d_model | d_ff | d_attn | val seq_acc | val tok_acc | val CE |
|---------|------|--------|-------------|-------------|--------|
| 16      | 64   | 16     | 0.56        | 0.97        | 0.15   |
| 32      | 128  | 32     | 0.57        | 0.97        | 0.17   |
| **64**  | **256** | **64** | **0.74** | **0.98** | **0.08** |
| 128     | 512  | 128    | 0.65        | 0.98        | 0.11   |
| 256     | 1024 | 256    | 0.57        | 0.97        | 0.16   |

## Depth Ablation (med width 32/128/32, baseline)

| Layers | val seq_acc | val tok_acc | val CE |
|--------|-------------|-------------|--------|
| 1      | 0.39        | 0.95        | 0.45   |
| **2**  | **0.67**    | **0.97**    | **0.16** |
| 4      | 0.57        | 0.97        | 0.17   |
| 8      | 0.54        | 0.96        | 0.19   |

## Sparsity Test (8 layers, d_model=64)

|                | softmax  | baseline |
|----------------|----------|----------|
| val seq_acc    | **0.81** | 0.49     |
| val tok_acc    | 0.99     | 0.96     |
| val CE         | 0.05     | 0.20     |
| d_eff          | 1.42     | —        |

Softmax layer sparsity prunes 8 layers to ~1.5 effective sublayers, recovering performance that baseline loses to overfitting.

## Widthwise Sparsity Test (4 layers, d_model=128)

|                | widthwise | baseline |
|----------------|-----------|----------|
| val seq_acc    | **0.89**  | 0.65     |
| val tok_acc    | 0.99      | 0.98     |
| val CE         | 0.03      | 0.11     |
| d_eff          | 38.3      | —        |

Baseline at d128 is in the overfitting regime (0.65 vs 0.74 at d64 sweet spot). Widthwise sparsity recovers and surpasses it — 0.89, the best val seq_acc across all experiments. Focus vectors effectively narrow the model's usable width, counteracting overparameterization.

Data: runs/widthwise_test_L4_d128_vs_baseline/

## Combined Sparsity — Perfect Generalization (4 layers, d_model=128, reg_examples=32)

|                | combined  | baseline |
|----------------|-----------|----------|
| val seq_acc    | **1.00**  | 0.15     |
| val tok_acc    | 1.00      | 0.90     |
| val CE         | 0.002     | 0.25     |
| train seq_acc  | 1.00      | 0.34     |
| d_eff          | 38.1      | —        |

Combined (layer + width sparsity) with stronger regularization achieves perfect generalization on all 192 held-out 8-bit binary increment examples, training on only 64 (25% of the space). Baseline fails to even memorize the training set under the same regularization strength.

Data: runs/combined_perfect_generalization_L4_d128_reg32/

---

## Summary: What We Learned (2026-03-22)

### The Problem
Standard transformers exhibit an inverted-U relationship between model size and data efficiency. Both width and depth have a sweet spot — go past it and generalization degrades due to overfitting on limited data.

### Width Ablation
Baseline at 4 layers: d_model 16→32→64→128→256 gives val_seq_acc 0.56→0.57→**0.74**→0.65→0.57. Sweet spot at d64, degradation beyond.

### Depth Ablation
Baseline at med width: Layers 1→2→4→8 gives val_seq_acc 0.39→**0.67**→0.57→0.54. Sweet spot at 2 layers, degradation beyond.

### Layerwise Sparsity (softmax)
At 8 layers d64 (where baseline gets 0.49), softmax layer sparsity recovers to **0.81** by pruning to d_eff=1.42 effective sublayers. Solves the depth overfitting problem.

### Widthwise Sparsity (focus vectors)
At 4 layers d128 (where baseline gets 0.65), widthwise sparsity reaches **0.89** with d_eff=38.3. Focus vectors learn which dimensions matter, solving the width overfitting problem. Key design:
- FocusNorm: `g ⊙ x / √(softmax(λ) · x²)` replaces RMSNorm
- Focus: `(softmax(λ)/||softmax(λ)||₂) ⊙ x` before each matmul replaces 1/√D scaling
- FocusNorm λ shared with first matmul's focus in each sublayer
- Focus on keys in d_attn space before Q·Kᵀ replaces 1/√A

### Combined (layer + width sparsity)
Combining both with stronger regularization (reg_examples=32) achieves **perfect generalization** — 100% val sequence accuracy on all 192 unseen examples from 64 training examples.

### Key Insight
Learnable sparsity (both layerwise and widthwise) decouples model capacity from data efficiency. You can scale the model freely and the sparsity mechanism learns the right effective size for the data, acting as an adaptive regularizer. This breaks the classical bias-variance tradeoff where model size must be carefully matched to dataset size.
