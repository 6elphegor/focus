#!/usr/bin/env python3
"""Plot width and depth ablation results."""

import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def get_final_val_seq_acc(run_dir):
    """Extract final val_seq_acc from a run directory."""
    for arch_dir in sorted(run_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        jsonl = arch_dir / "metrics_live.jsonl"
        if jsonl.exists():
            last = None
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
            if last:
                val = last.get("val_seq_acc", float("nan"))
                if not math.isnan(val):
                    return val
    return float("nan")

# Width ablation
width_runs = [
    ("16",  "runs/ablation_width/d16_ff64_a16"),
    ("32",  "runs/ablation_width/d32_ff128_a32"),
    ("64",  "runs/ablation_width/d64_ff256_a64"),
    ("128", "runs/ablation_width/d128_ff512_a128"),
    ("256", "runs/ablation_width/d256_ff1024_a256"),
]

# Depth ablation
depth_runs = [
    ("1", "runs/ablation_depth/L1"),
    ("2", "runs/ablation_depth/L2"),
    ("4", "runs/ablation_depth/L4"),
    ("8", "runs/ablation_depth/L8"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#f8f8f8")

# Width plot
labels_w = [r[0] for r in width_runs]
vals_w = [get_final_val_seq_acc(Path(r[1])) for r in width_runs]
ax1.plot(labels_w, vals_w, "o-", color="#1f77b4", linewidth=2, markersize=8)
ax1.set_title("Width Ablation (4 layers, 5k iters)", fontsize=13, fontweight="bold")
ax1.set_xlabel("d_model")
ax1.set_ylabel("Val Sequence Accuracy")
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)
for i, v in enumerate(vals_w):
    ax1.annotate(f"{v:.2f}", (labels_w[i], v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

# Depth plot
labels_d = [r[0] for r in depth_runs]
vals_d = [get_final_val_seq_acc(Path(r[1])) for r in depth_runs]
ax2.plot(labels_d, vals_d, "o-", color="#ff7f0e", linewidth=2, markersize=8)
ax2.set_title("Depth Ablation (med width 32/128/32, 5k iters)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Layers")
ax2.set_ylabel("Val Sequence Accuracy")
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
for i, v in enumerate(vals_d):
    ax2.annotate(f"{v:.2f}", (labels_d[i], v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("runs/ablation_plots.png", dpi=150)
plt.show()
