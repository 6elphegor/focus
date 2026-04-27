#!/usr/bin/env python3
"""Live training monitor GUI. Polls metrics_live.jsonl or metrics.json and updates plots."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

COLORS = {
    "canonical_baseline": "#1f77b4",
    "canonical_softmax": "#ff7f0e",
    "canonical_widthwise": "#2ca02c",
    "canonical_combined": "#d62728",
}

ALL_FIELDS = [
    "step", "full_loss", "ce_loss", "token_acc", "seq_acc", "d_eff",
    "val_ce_loss", "val_token_acc", "val_seq_acc",
]


def load_metrics(arch_dir: Path) -> dict:
    """Load metrics from jsonl (preferred) or json fallback."""
    data = {k: [] for k in ALL_FIELDS}
    data["d_eff_detail"] = None  # last d_eff_detail dict if present

    jsonl = arch_dir / "metrics_live.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for k in ALL_FIELDS:
                    data[k].append(row.get(k, float("nan")))
                if "d_eff_detail" in row:
                    data["d_eff_detail"] = row["d_eff_detail"]
        return data

    # Fallback: read metrics.json (written at end of training)
    json_path = arch_dir / "metrics.json"
    if json_path.exists():
        with open(json_path) as f:
            m = json.load(f)
        data["step"] = m.get("steps", [])
        for k in ALL_FIELDS:
            if k != "step" and k in m:
                data[k] = m[k]
        return data

    # Last resort: scan checkpoint files for embedded metrics
    ckpts = sorted(arch_dir.glob("ckpt_*.pt"))
    if ckpts:
        import torch
        ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        m = ckpt.get("metrics", {})
        data["step"] = m.get("steps", [])
        for k in ALL_FIELDS:
            if k != "step" and k in m:
                data[k] = m[k]

    return data


def filter_nan(steps, values):
    """Return only points where value is not NaN."""
    s, v = [], []
    for si, vi in zip(steps, values):
        if not math.isnan(vi):
            s.append(si)
            v.append(vi)
    return s, v


def main():
    parser = argparse.ArgumentParser(description="Live training monitor")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--interval", type=int, default=3000, help="Poll interval in ms")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        runs_root = Path("runs")
        if runs_root.exists():
            candidates = [d for d in runs_root.iterdir() if d.is_dir()]
            if candidates:
                out_dir = max(candidates, key=lambda d: d.stat().st_mtime)
            else:
                out_dir = runs_root
        else:
            out_dir = runs_root
    print(f"Monitoring: {out_dir}")

    def find_archs():
        archs = []
        if out_dir.exists():
            for d in sorted(out_dir.iterdir()):
                if d.is_dir():
                    archs.append(d.name)
        return archs

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Training Monitor", fontsize=16, fontweight="bold")
    fig.patch.set_facecolor("#f8f8f8")
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.92)

    # (field, title, axis, is_accuracy, val_field_or_None)
    panel_cfg = [
        ("full_loss",  "Full Loss",         axes[0, 0], False, None),
        ("ce_loss",    "Cross-Entropy Loss", axes[0, 1], False, "val_ce_loss"),
        ("d_eff",      "d_eff",              axes[0, 2], False, None),
        ("token_acc",  "Token Accuracy",     axes[1, 0], True,  "val_token_acc"),
        ("seq_acc",    "Sequence Accuracy",  axes[1, 1], True,  "val_seq_acc"),
    ]

    # Status panels
    status_ax = axes[0, 3]
    status_ax.axis("off")
    val_status_ax = axes[1, 2]
    val_status_ax.axis("off")
    empty_ax = axes[1, 3]
    empty_ax.axis("off")

    def update(frame):
        archs = find_archs()
        all_data = {}
        for arch in archs:
            all_data[arch] = load_metrics(out_dir / arch)

        for key, title, ax, is_acc, val_key in panel_cfg:
            ax.clear()
            ax.set_facecolor("#fdfdfd")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("step")
            if is_acc:
                ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            for arch in archs:
                d = all_data[arch]
                if d["step"]:
                    color = COLORS.get(arch, None)
                    ax.plot(d["step"], d[key], label=f"{arch} (train)", color=color, linewidth=1.5)
                    if val_key and d.get(val_key):
                        vs, vv = filter_nan(d["step"], d[val_key])
                        if vs:
                            ax.plot(vs, vv, label=f"{arch} (val)", color=color, linewidth=1.5, linestyle="--")
            if any(all_data.get(a, {}).get("step") for a in archs):
                ax.legend(fontsize=8)

        # Train status panel
        status_ax.clear()
        status_ax.axis("off")
        status_ax.set_facecolor("#fdfdfd")
        lines = []
        for arch in archs:
            d = all_data[arch]
            if d["step"]:
                lines.append(f"--- {arch} ---")
                lines.append(f"  step:    {d['step'][-1]}")
                lines.append(f"  loss:    {d['full_loss'][-1]:.4f}")
                lines.append(f"  ce:      {d['ce_loss'][-1]:.4f}")
                lines.append(f"  tok_acc: {d['token_acc'][-1]:.4f}")
                lines.append(f"  seq_acc: {d['seq_acc'][-1]:.4f}")
                lines.append(f"  d_eff:   {d['d_eff'][-1]:.2f}")
                if d.get("d_eff_detail"):
                    for name, val in d["d_eff_detail"].items():
                        short = name.replace("blocks.", "B").replace(".slide_attn", ".sa").replace(".full_attn", ".fa").replace(".ffn1", ".f1").replace(".ffn2", ".f2").replace("focus_", "f_").replace("norm_slide", "ns").replace("norm_ffn1", "nf1").replace("norm_full", "nf").replace("norm_ffn2", "nf2").replace("focus_out", "f_out")
                        lines.append(f"    {short}: {val:.1f}")
                lines.append("")
        if not lines:
            lines.append("Waiting for data...")
            lines.append(f"Watching: {out_dir}")
        status_ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=status_ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top",
        )
        status_ax.set_title("Train (latest)", fontsize=12, fontweight="bold")

        # Val status panel
        val_status_ax.clear()
        val_status_ax.axis("off")
        val_status_ax.set_facecolor("#fdfdfd")
        vlines = []
        for arch in archs:
            d = all_data[arch]
            vce = [v for v in d.get("val_ce_loss", []) if not math.isnan(v)]
            vtok = [v for v in d.get("val_token_acc", []) if not math.isnan(v)]
            vseq = [v for v in d.get("val_seq_acc", []) if not math.isnan(v)]
            if vce:
                vlines.append(f"--- {arch} ---")
                vlines.append(f"  val_ce:      {vce[-1]:.4f}")
                vlines.append(f"  val_tok_acc: {vtok[-1]:.4f}")
                vlines.append(f"  val_seq_acc: {vseq[-1]:.4f}")
                vlines.append("")
        if not vlines:
            vlines.append("No val data yet")
        val_status_ax.text(
            0.05, 0.95, "\n".join(vlines),
            transform=val_status_ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top",
        )
        val_status_ax.set_title("Val (latest)", fontsize=12, fontweight="bold")

        empty_ax.clear()
        empty_ax.axis("off")

    ani = FuncAnimation(fig, update, interval=args.interval, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
