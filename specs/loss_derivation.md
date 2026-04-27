# Training Notes

## Assumptions

- All training tokens are processed in a single batch.
- A prior distribution is defined over all model weights. This includes token embeddings and RMS norm elementwise scaling parameters.

## Bayesian Loss Derivation

By Bayes' rule with fixed data:

```
p(params | data) ∝ p(data | params) · p(params)
```

Taking the negative log and dropping constants (which vanish under gradients):

```
loss ∝ -ln p(data | params) - ln p(params)
```

The term `-ln p(params)` is the information content of the weights in nats. For small precisions, `p(params) ≈ pdf(params) · ∏ᵢ precision_wᵢ`, so `-ln p(params) = -ln pdf(params) - ∑ᵢ ln precision_wᵢ`. The precision terms are constants that vanish under gradients and can be excluded from the loss. Normalizing by the total number of training tokens `L = ∑ᵢ |data[i]|`:

```
loss = (1/L) · (-ln p(data | params) - ln pdf(params))
       ╰──────── cross entropy ────────╯   ╰─ regularization ─╯
```

Note that dividing the regularization term by `L` means regularization weakens as the dataset grows. In the limit of infinite data, no regularization is needed — which is correct when the entire token distribution is observed.

**Do not use standard weight decay from ML libraries.** The full loss above handles regularization properly.

## Training Protocol

- Use AdamW as the optimizer.
- Do not train for a fixed number of iterations. Training stops when either the user decides to stop (based on observing the full loss plateau) or the loss reaches a lower bound.
- Display two curves during training:
  1. **Cross entropy loss:** `(1/L) · (-ln p(data | params))` — measures prediction quality.
  2. **Full loss:** cross entropy + regularization — a more holistic measure accounting for model complexity.
- On stop: save a plot of both curves.
- Support resuming training if the user decides more steps are needed after inspecting the plot.
- After the user stops training, run evaluation on the test set.

## Saved Artifacts

- **Training loss log:** Save per-iteration cross entropy loss and full loss to a file (e.g., CSV with columns `iteration, cross_entropy_loss, full_loss`).
- **Model hyperparameters:** Save all model hyperparameters to a file (e.g., JSON) before training begins.
- **Test results:** Save evaluation results on the test set to a file after the user stops training.
