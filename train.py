#!/usr/bin/env python3
"""Canonical Baseline vs Canonical Softmax on 16-bit binary increment task."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# AdamW with per-parameter (not per-scalar) second moment
# ---------------------------------------------------------------------------
class AdamWPerParam(torch.optim.Optimizer):
    """AdamW where the second moment v is a single scalar per parameter tensor."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)

                state["step"] += 1
                m, v = state["m"], state["v"]

                # First moment: per-element (preserves direction)
                m.mul_(b1).add_(g, alpha=1 - b1)
                # Second moment: single scalar = mean of squared gradients
                v.mul_(b2).add_((g * g).mean(), alpha=1 - b2)

                # Bias correction
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                m_hat = m / bc1
                v_hat = v / bc2

                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XI = 0.596469211  # sqrt(E[silu(z)^2]) for z ~ N(0,1)
VOCAB = ["0", "1", "→", "C", "NC"]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
V = len(VOCAB)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def bits_to_sequence(bits: list[int], num_bits: int) -> list[int]:
    """Convert LSB-first bit list to full increment sequence token ids."""
    tokens = [str(b) for b in bits]
    tokens.append("→")
    carry = 1
    for i in range(num_bits):
        s = bits[i] + carry
        y_i = s % 2
        carry = s // 2
        tokens.append(str(y_i))
        tokens.append("C" if carry else "NC")
    return [TOK2ID[t] for t in tokens]


def make_batch(batch_size: int, num_bits: int, split_seed: int, step: int) -> torch.Tensor:
    """Generate a batch of binary-increment sequences deterministically."""
    seed = (split_seed << 32) + step
    g = torch.Generator()
    g.manual_seed(seed)
    xs = torch.randint(0, 2**num_bits, (batch_size,), generator=g)

    seqs = []
    for x_val in xs.tolist():
        bits = [(x_val >> i) & 1 for i in range(num_bits)]
        seqs.append(bits_to_sequence(bits, num_bits))

    return torch.tensor(seqs, dtype=torch.long)


def load_fixed_dataset(path: str, num_bits: int) -> tuple[torch.Tensor, set[int]]:
    """Load a fixed dataset of LSB-first binary strings. Returns (sequences, set of integer values)."""
    seqs = []
    values = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            assert len(line) == num_bits, f"Expected {num_bits}-bit string, got {len(line)}: {line}"
            bits = [int(c) for c in line]
            # Convert LSB-first bits to integer
            val = sum(b << i for i, b in enumerate(bits))
            values.add(val)
            seqs.append(bits_to_sequence(bits, num_bits))
    return torch.tensor(seqs, dtype=torch.long), values


def build_complement_dataset(train_values: set[int], num_bits: int) -> torch.Tensor:
    """Build validation set from all num_bits-bit integers NOT in train_values."""
    seqs = []
    for x in range(2**num_bits):
        if x not in train_values:
            bits = [(x >> i) & 1 for i in range(num_bits)]
            seqs.append(bits_to_sequence(bits, num_bits))
    return torch.tensor(seqs, dtype=torch.long)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(ms)
        return (x * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# FocusNorm (widthwise sparsity)
# ---------------------------------------------------------------------------
class Focus(nn.Module):
    """Learnable focus vector: f = softmax(lam) / ||softmax(lam)||_2."""
    def __init__(self, d: int):
        super().__init__()
        self.lam = nn.Parameter(torch.zeros(d))

    def focus(self) -> torch.Tensor:
        s = F.softmax(self.lam, dim=0)
        return s / s.norm(p=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.focus()


class FocusNorm(nn.Module):
    """RMSNorm variant with learnable focus. Shares lambda with a Focus instance.
    norm: s = softmax(lam) (sums to 1), x_norm = weight * x / sqrt(s dot x^2)
    The shared Focus uses f = softmax(lam) / ||softmax(lam)||_2 for hadamard."""
    def __init__(self, d: int):
        super().__init__()
        self.lam = nn.Parameter(torch.zeros(d))
        self.weight = nn.Parameter(torch.ones(d))

    def focus(self) -> torch.Tensor:
        """Focus vector (L2-normalized softmax) for hadamard before first matmul."""
        s = F.softmax(self.lam, dim=0)
        return s / s.norm(p=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.softmax(self.lam, dim=0)
        # Weighted RMS: sqrt(sum(s * x^2)), s sums to 1
        wms = (s * x.float().pow(2)).sum(-1, keepdim=True)
        x = x.float() * torch.rsqrt(wms + 1e-8)
        return (x * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len: int, d: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE. d is the head dim."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # [seq, d//2]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings. x: [batch, seq, d]"""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    cos = cos[: x.shape[1]]
    sin = sin[: x.shape[1]]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, d_model: int, d_attn: int, sliding: bool, window_size: int):
        super().__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.sliding = sliding
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.tau = nn.Parameter(torch.ones(()))

    def forward(self, h: torch.Tensor, rope_cos: torch.Tensor | None, rope_sin: torch.Tensor | None) -> torch.Tensor:
        B, T, D = h.shape
        A = self.d_attn
        scale = math.sqrt(D)

        Q = self.W_q(h) / scale  # [B, T, A]
        K = self.W_k(h) / scale
        V = self.W_v(h) / scale

        if self.sliding and rope_cos is not None:
            Q = apply_rope(Q, rope_cos, rope_sin)
            K = apply_rope(K, rope_cos, rope_sin)

        # Scores: tau * (Q K^T) / sqrt(A), then squared
        scores = self.tau * torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(A)
        scores = scores.square()

        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))

        # Sliding window mask
        if self.sliding:
            # mask out keys more than window_size positions in the past
            positions = torch.arange(T, device=h.device)
            dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # [T, T] query_pos - key_pos
            window_mask = dist > self.window_size
            scores = scores.masked_fill(window_mask.unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)


# ---------------------------------------------------------------------------
# Attention (widthwise sparsity)
# ---------------------------------------------------------------------------
class AttentionWidth(nn.Module):
    """Attention with widthwise focus vectors before each matmul.
    focus_q is shared with the sublayer's FocusNorm (set after construction)."""
    def __init__(self, d_model: int, d_attn: int, sliding: bool, window_size: int):
        super().__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.sliding = sliding
        self.window_size = window_size

        # f_q, f_k, f_v all shared with FocusNorm — set via set_shared_focus()
        self._shared_focus_norm = None
        # Focus on keys in d_attn space before Q·K^T (replaces 1/sqrt(A))
        self.focus_k_attn = Focus(d_attn)

        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.tau = nn.Parameter(torch.ones(()))

    def set_shared_focus(self, norm: 'FocusNorm'):
        self._shared_focus_norm = norm

    def forward(self, h: torch.Tensor, rope_cos: torch.Tensor | None, rope_sin: torch.Tensor | None) -> torch.Tensor:
        B, T, D = h.shape

        # Q, K, V all use shared focus from FocusNorm
        f = self._shared_focus_norm.focus()
        Q = self.W_q(h * f)
        K = self.W_k(h * f)
        V = self.W_v(h * f)

        if self.sliding and rope_cos is not None:
            Q = apply_rope(Q, rope_cos, rope_sin)
            K = apply_rope(K, rope_cos, rope_sin)

        # Focus keys in attn space before dot product (replaces 1/sqrt(A))
        K = self.focus_k_attn(K)

        # Scores: tau * (Q K^T), then squared
        scores = self.tau * torch.bmm(Q, K.transpose(1, 2))
        scores = scores.square()

        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))

        # Sliding window mask
        if self.sliding:
            positions = torch.arange(T, device=h.device)
            dist = positions.unsqueeze(0) - positions.unsqueeze(1)
            window_mask = dist > self.window_size
            scores = scores.masked_fill(window_mask.unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)


# ---------------------------------------------------------------------------
# FFN (SwiGLU) widthwise sparsity
# ---------------------------------------------------------------------------
class FFNWidth(nn.Module):
    """FFN with widthwise focus vectors before each matmul.
    focus_gate is shared with the sublayer's FocusNorm (set after construction)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # focus_gate and focus_up shared with FocusNorm — set via set_shared_focus()
        self._shared_focus_norm = None
        self.focus_down = Focus(d_ff)
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def set_shared_focus(self, norm: 'FocusNorm'):
        self._shared_focus_norm = norm

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        f = self._shared_focus_norm.focus()
        gate = self.W_gate(h * f)
        up = self.W_up(h * f)
        mix = F.silu(gate) * up
        return self.W_down(self.focus_down(mix))


# ---------------------------------------------------------------------------
# FFN (SwiGLU)
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scale_in = math.sqrt(self.d_model)
        gate = self.W_gate(h) / scale_in
        up = self.W_up(h) / scale_in
        mix = F.silu(gate) * up
        return self.W_down(mix) / (XI * math.sqrt(self.d_ff))


# ---------------------------------------------------------------------------
# Block pair
# ---------------------------------------------------------------------------
class BlockPair(nn.Module):
    def __init__(self, d_model: int, d_attn: int, d_ff: int, window_size: int):
        super().__init__()
        # Sliding attention sublayer
        self.norm_slide = RMSNorm(d_model)
        self.slide_attn = Attention(d_model, d_attn, sliding=True, window_size=window_size)

        # FFN 1
        self.norm_ffn1 = RMSNorm(d_model)
        self.ffn1 = FFN(d_model, d_ff)

        # Full attention sublayer
        self.norm_full = RMSNorm(d_model)
        self.full_attn = Attention(d_model, d_attn, sliding=False, window_size=window_size)

        # FFN 2
        self.norm_ffn2 = RMSNorm(d_model)
        self.ffn2 = FFN(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        s0, s1, s2, s3 = scales
        x = x + s0 * self.slide_attn(self.norm_slide(x), rope_cos, rope_sin)
        x = x + s1 * self.ffn1(self.norm_ffn1(x))
        x = x + s2 * self.full_attn(self.norm_full(x), None, None)
        x = x + s3 * self.ffn2(self.norm_ffn2(x))
        return x


# ---------------------------------------------------------------------------
# Block pair (widthwise sparsity)
# ---------------------------------------------------------------------------
class BlockPairWidth(nn.Module):
    def __init__(self, d_model: int, d_attn: int, d_ff: int, window_size: int):
        super().__init__()
        self.norm_slide = FocusNorm(d_model)
        self.slide_attn = AttentionWidth(d_model, d_attn, sliding=True, window_size=window_size)
        self.norm_ffn1 = FocusNorm(d_model)
        self.ffn1 = FFNWidth(d_model, d_ff)
        self.norm_full = FocusNorm(d_model)
        self.full_attn = AttentionWidth(d_model, d_attn, sliding=False, window_size=window_size)
        self.norm_ffn2 = FocusNorm(d_model)
        self.ffn2 = FFNWidth(d_model, d_ff)

        # Share FocusNorm lambda with first focus in each sublayer
        self.slide_attn.set_shared_focus(self.norm_slide)
        self.ffn1.set_shared_focus(self.norm_ffn1)
        self.full_attn.set_shared_focus(self.norm_full)
        self.ffn2.set_shared_focus(self.norm_ffn2)

    def forward(
        self,
        x: torch.Tensor,
        scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        s0, s1, s2, s3 = scales
        x = x + s0 * self.slide_attn(self.norm_slide(x), rope_cos, rope_sin)
        x = x + s1 * self.ffn1(self.norm_ffn1(x))
        x = x + s2 * self.full_attn(self.norm_full(x), None, None)
        x = x + s3 * self.ffn2(self.norm_ffn2(x))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CanonicalModel(nn.Module):
    def __init__(self, arch: str, d_model: int, d_attn: int, d_ff: int, num_layers: int, window_size: int):
        super().__init__()
        self.arch = arch
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_sublayers = 4 * num_layers

        self.E_tok = nn.Embedding(V, d_model)
        self.W_out = nn.Linear(d_model, V, bias=False)

        if arch in ("canonical_widthwise", "canonical_combined"):
            self.blocks = nn.ModuleList([
                BlockPairWidth(d_model, d_attn, d_ff, window_size) for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                BlockPair(d_model, d_attn, d_ff, window_size) for _ in range(num_layers)
            ])

        if arch == "canonical_baseline":
            self.s = nn.Parameter(torch.ones(self.num_sublayers))
        elif arch == "canonical_softmax":
            self.lam = nn.Parameter(torch.zeros(self.num_sublayers))
            self.s = nn.Parameter(torch.ones(self.num_sublayers))
        elif arch == "canonical_widthwise":
            self.s = nn.Parameter(torch.ones(self.num_sublayers))
            self.focus_out = Focus(d_model)
        elif arch == "canonical_combined":
            # Layer sparsity (from softmax)
            self.lam = nn.Parameter(torch.zeros(self.num_sublayers))
            self.s = nn.Parameter(torch.ones(self.num_sublayers))
            # Width sparsity (from widthwise)
            self.focus_out = Focus(d_model)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def _get_scales(self) -> torch.Tensor:
        """Return per-sublayer scales [num_sublayers]."""
        if self.arch in ("canonical_baseline", "canonical_widthwise"):
            return self.s
        elif self.arch in ("canonical_softmax", "canonical_combined"):
            p = F.softmax(self.lam, dim=0)
            alpha = p / p.norm(p=2)
            return self.s * alpha
        return self.s

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        B, T = x_ids.shape
        x = self.E_tok(x_ids)  # [B, T, D]

        rope_cos, rope_sin = build_rope_cache(T, self.blocks[0].slide_attn.d_attn, x.device)
        scales = self._get_scales()

        for i, block in enumerate(self.blocks):
            block_scales = (scales[4*i], scales[4*i+1], scales[4*i+2], scales[4*i+3])
            x = block(x, block_scales, rope_cos, rope_sin)

        # Output projection
        if self.arch in ("canonical_widthwise", "canonical_combined"):
            logits = self.W_out(self.focus_out(x))
        elif self.arch == "canonical_baseline":
            denom = math.sqrt(self.d_model * (1 + self.num_sublayers))
            logits = self.W_out(x) / denom
        elif self.arch == "canonical_softmax":
            denom = math.sqrt(2.0 * self.d_model)
            logits = self.W_out(x) / denom
        return logits

    def weight_prior_loss(self) -> torch.Tensor:
        """0.5 * sum of all trainable params squared (architecture-dependent exclusions)."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, p in self.named_parameters():
            if self.arch in ("canonical_softmax", "canonical_combined") and name == "lam":
                continue
            if self.arch in ("canonical_widthwise", "canonical_combined") and name.endswith(".lam") and name != "lam":
                continue
            total = total + p.float().pow(2).sum()
        return 0.5 * total

    def d_eff(self) -> torch.Tensor:
        """Effective dimensionality. 0 for baseline, 1/sum(alpha^4) for softmax/widthwise."""
        if self.arch == "canonical_baseline":
            return torch.tensor(0.0, device=next(self.parameters()).device)
        elif self.arch == "canonical_softmax":
            p = F.softmax(self.lam, dim=0)
            alpha = p / p.norm(p=2)
            return 1.0 / (alpha ** 4).sum()
        elif self.arch in ("canonical_widthwise", "canonical_combined"):
            total = torch.tensor(0.0, device=next(self.parameters()).device)
            # Layer sparsity d_eff (combined only)
            if self.arch == "canonical_combined":
                p = F.softmax(self.lam, dim=0)
                alpha = p / p.norm(p=2)
                total = total + 1.0 / (alpha ** 4).sum()
            # Width sparsity d_effs
            for m in self.modules():
                if isinstance(m, FocusNorm):
                    f = m.focus()
                    total = total + 1.0 / (f ** 4).sum()
                elif isinstance(m, Focus):
                    f = m.focus()
                    total = total + 1.0 / (f ** 4).sum()
            return total
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def d_eff_detail(self) -> dict[str, float]:
        """Per-FocusNorm d_eff values. Only for canonical_widthwise."""
        if self.arch not in ("canonical_widthwise", "canonical_combined"):
            return {}
        result = {}
        for name, m in self.named_modules():
            if isinstance(m, FocusNorm):
                with torch.no_grad():
                    f = m.focus()
                    result[name] = (1.0 / (f ** 4).sum()).item()
            elif isinstance(m, Focus):
                with torch.no_grad():
                    f = m.focus()
                    result[name] = (1.0 / (f ** 4).sum()).item()
        return result


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def _is_special_param(name: str) -> bool:
    """Check if a param gets deterministic (non-PRNG) initialization."""
    if "norm" in name and "weight" in name:
        return True
    if name.endswith(".tau"):
        return True
    if name == "s":
        return True
    if name == "lam" or name.endswith(".lam"):
        return True
    # FocusNorm focus weights (inside attention/ffn) — init to ones
    if "focus" in name and "weight" in name:
        return True
    return False


# Canonical param order: the iteration order of a baseline model's random-inited params.
# This ensures all architectures consume PRNG in the same order for shared weights.
_BASELINE_PARAM_ORDER = None


def _get_baseline_param_order(d_model, d_attn, d_ff, num_layers, window_size):
    global _BASELINE_PARAM_ORDER
    ref = CanonicalModel("canonical_baseline", d_model, d_attn, d_ff, num_layers, window_size)
    _BASELINE_PARAM_ORDER = [n for n, _ in ref.named_parameters() if not _is_special_param(n)]
    return _BASELINE_PARAM_ORDER


def init_params(model: CanonicalModel, init_seed: int):
    """Deterministic initialization: Uniform(-sqrt(3), sqrt(3)) except special cases.
    Shared weights get the same PRNG values regardless of architecture."""
    g = torch.Generator(device="cpu")
    g.manual_seed(init_seed)

    params = dict(model.named_parameters())

    # First: init all special params deterministically (no PRNG)
    for name, p in params.items():
        if not _is_special_param(name):
            continue
        if name.endswith(".lam"):
            nn.init.zeros_(p)
        else:
            nn.init.ones_(p)

    # Second: init shared (baseline-order) params using PRNG in canonical order
    baseline_order = _get_baseline_param_order(
        model.d_model,
        model.blocks[0].slide_attn.d_attn if hasattr(model.blocks[0].slide_attn, 'd_attn') else model.blocks[0].slide_attn.d_attn,
        model.blocks[0].ffn1.d_ff,
        model.num_layers,
        model.blocks[0].slide_attn.window_size,
    )
    for name in baseline_order:
        if name in params:
            p = params[name]
            val = torch.empty_like(p, device="cpu").uniform_(-math.sqrt(3), math.sqrt(3), generator=g)
            p.data.copy_(val.to(p.device))

    # Third: init any remaining non-special params (widthwise-only random params) with continued PRNG
    initialized = set(baseline_order)
    for name, p in params.items():
        if _is_special_param(name) or name in initialized:
            continue
        val = torch.empty_like(p, device="cpu").uniform_(-math.sqrt(3), math.sqrt(3), generator=g)
        p.data.copy_(val.to(p.device))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_plots(metrics: dict, out_dir: Path, show: bool):
    """Save training curves as PNG."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    for arch_name, arch_metrics in metrics.items():
        if not arch_metrics["steps"]:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(arch_name, fontsize=14)

        steps = arch_metrics["steps"]

        axes[0, 0].plot(steps, arch_metrics["full_loss"], label="full_loss")
        axes[0, 0].plot(steps, arch_metrics["ce_loss"], label="ce_loss")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("step")

        axes[0, 1].plot(steps, arch_metrics["token_acc"], label="token_acc")
        axes[0, 1].set_title("Token Accuracy")
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("step")
        axes[0, 1].set_ylim(0, 1)

        axes[1, 0].plot(steps, arch_metrics["seq_acc"], label="seq_acc")
        axes[1, 0].set_title("Sequence Accuracy")
        axes[1, 0].legend()
        axes[1, 0].set_xlabel("step")
        axes[1, 0].set_ylim(0, 1)

        axes[1, 1].plot(steps, arch_metrics["d_eff"], label="d_eff")
        axes[1, 1].set_title("d_eff")
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("step")

        plt.tight_layout()
        fig.savefig(out_dir / f"{arch_name}.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    # Combined comparison plot
    if len(metrics) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Comparison", fontsize=14)
        for arch_name, arch_metrics in metrics.items():
            if not arch_metrics["steps"]:
                continue
            steps = arch_metrics["steps"]
            axes[0, 0].plot(steps, arch_metrics["full_loss"], label=arch_name)
            axes[0, 1].plot(steps, arch_metrics["token_acc"], label=arch_name)
            axes[1, 0].plot(steps, arch_metrics["seq_acc"], label=arch_name)
            axes[1, 1].plot(steps, arch_metrics["d_eff"], label=arch_name)

        for ax, title in zip(axes.flat, ["Full Loss", "Token Accuracy", "Sequence Accuracy", "d_eff"]):
            ax.set_title(title)
            ax.legend()
            ax.set_xlabel("step")
        axes[0, 1].set_ylim(0, 1)
        axes[1, 0].set_ylim(0, 1)

        plt.tight_layout()
        fig.savefig(out_dir / "comparison.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_arch(
    arch: str,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Training: {arch}")
    print(f"{'='*60}")

    model = CanonicalModel(
        arch=arch,
        d_model=args.d_model,
        d_attn=args.d_attn,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        window_size=args.attention_window_size,
    ).to(device)
    init_params(model, args.init_seed)

    # Print param count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    if args.compile_mode:
        model = torch.compile(model, mode=args.compile_mode)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=0.0,  # we handle regularization manually
        )
    elif args.optimizer == "adamw_perparam":
        optimizer = AdamWPerParam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # LR schedule
    scheduler = None
    if args.lr_schedule == "cosine":
        warmup_steps = int(args.warmup_fraction * args.train_iterations)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(args.train_iterations - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = args.min_lr / args.learning_rate
            return min_ratio + (1.0 - min_ratio) * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    num_bits = args.num_bits
    prompt_len = num_bits + 1  # bits + arrow
    output_len = 2 * num_bits
    loss_tokens = args.regularizer_examples * output_len  # 65536 * 32

    metrics = {
        "steps": [], "full_loss": [], "ce_loss": [], "token_acc": [], "seq_acc": [], "d_eff": [],
        "val_ce_loss": [], "val_token_acc": [], "val_seq_acc": [],
    }
    arch_dir = out_dir / arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    # Load fixed dataset if provided
    fixed_batch = None
    val_batch = None
    if args.fixed_train_file:
        fixed_batch, train_values = load_fixed_dataset(args.fixed_train_file, num_bits)
        fixed_batch = fixed_batch.to(device)
        val_batch = build_complement_dataset(train_values, num_bits).to(device)
        print(f"Fixed dataset: {fixed_batch.shape[0]} train, {val_batch.shape[0]} val from {args.fixed_train_file}")

    t_start = time.time()

    for step in range(1, args.train_iterations + 1):
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            batch = make_batch(args.synthetic_train_batch_size, num_bits, args.split_seed, step).to(device)

        # Input: tokens 0..47, target: tokens 1..48
        inputs = batch[:, :-1]   # [B, 48]
        targets = batch[:, 1:]   # [B, 48]

        logits = model(inputs)   # [B, 48, V]

        # Loss only on supervised output tokens (after the arrow)
        # The arrow is at position prompt_len-1 = 16 in the original sequence
        # In the shifted target, output tokens start at position prompt_len-1 = 16
        output_logits = logits[:, prompt_len - 1 :, :]  # [B, 32, V]
        output_targets = targets[:, prompt_len - 1 :]    # [B, 32]

        ce_loss = F.cross_entropy(output_logits.reshape(-1, V), output_targets.reshape(-1))

        # Get underlying model for regularization
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        wpl = base_model.weight_prior_loss()
        d_eff_val = base_model.d_eff()

        full_loss = ce_loss + (wpl + d_eff_val) / loss_tokens

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Logging
        if step % args.plot_every == 0 or step == 1:
            with torch.no_grad():
                preds = output_logits.argmax(dim=-1)
                correct_tokens = (preds == output_targets).float().mean().item()
                correct_seqs = (preds == output_targets).all(dim=-1).float().mean().item()

            # Validation eval
            val_ce = val_tok = val_seq = float("nan")
            if val_batch is not None and step % args.validate_every == 0:
                with torch.no_grad():
                    val_inputs = val_batch[:, :-1]
                    val_targets = val_batch[:, 1:]
                    val_logits = model(val_inputs)
                    val_out_logits = val_logits[:, prompt_len - 1 :, :]
                    val_out_targets = val_targets[:, prompt_len - 1 :]
                    val_ce = F.cross_entropy(val_out_logits.reshape(-1, V), val_out_targets.reshape(-1)).item()
                    val_preds = val_out_logits.argmax(dim=-1)
                    val_tok = (val_preds == val_out_targets).float().mean().item()
                    val_seq = (val_preds == val_out_targets).all(dim=-1).float().mean().item()

            row = {
                "step": step,
                "full_loss": full_loss.item(),
                "ce_loss": ce_loss.item(),
                "token_acc": correct_tokens,
                "seq_acc": correct_seqs,
                "d_eff": d_eff_val.item(),
                "val_ce_loss": val_ce,
                "val_token_acc": val_tok,
                "val_seq_acc": val_seq,
            }
            # Log per-FocusNorm d_eff detail for widthwise arch
            d_eff_detail = base_model.d_eff_detail()
            if d_eff_detail:
                row["d_eff_detail"] = d_eff_detail
            metrics["steps"].append(row["step"])
            metrics["full_loss"].append(row["full_loss"])
            metrics["ce_loss"].append(row["ce_loss"])
            metrics["token_acc"].append(row["token_acc"])
            metrics["seq_acc"].append(row["seq_acc"])
            metrics["d_eff"].append(row["d_eff"])
            metrics["val_ce_loss"].append(row["val_ce_loss"])
            metrics["val_token_acc"].append(row["val_token_acc"])
            metrics["val_seq_acc"].append(row["val_seq_acc"])

            # Write incrementally for live monitoring
            with open(arch_dir / "metrics_live.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

            elapsed = time.time() - t_start
            val_str = ""
            if not math.isnan(val_ce):
                val_str = f" | val_ce {val_ce:.4f} | val_tok {val_tok:.4f} | val_seq {val_seq:.4f}"
            print(
                f"[{arch}] step {step:5d} | loss {full_loss.item():.4f} | "
                f"ce {ce_loss.item():.4f} | tok_acc {correct_tokens:.4f} | "
                f"seq_acc {correct_seqs:.4f} | d_eff {d_eff_val.item():.2f}{val_str} | "
                f"{elapsed:.1f}s"
            )

        # Checkpoint
        if step % args.checkpoint_every == 0:
            ckpt = {
                "step": step,
                "model_state": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
            }
            torch.save(ckpt, arch_dir / f"ckpt_{step:05d}.pt")

    # Save final metrics
    with open(arch_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+", default=["canonical_softmax", "canonical_baseline"])
    parser.add_argument("--out-dir", type=str, default="runs/compare")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--compile-mode", type=str, default=None)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--plot-every", type=int, default=50)
    parser.add_argument("--train-iterations", type=int, default=5000)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--num-bits", type=int, default=16)
    parser.add_argument("--attention-window-size", type=int, default=8)
    parser.add_argument("--synthetic-train", action="store_true")
    parser.add_argument("--synthetic-train-batch-size", type=int, default=16)
    parser.add_argument("--fixed-train-file", type=str, default=None, help="Path to fixed dataset file (LSB-first binary strings, one per line)")
    parser.add_argument("--disable-validation", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--d-attn", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    # Hidden default for regularizer scaling
    parser.add_argument("--regularizer-examples", type=int, default=65536)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine"],
                        help="LR schedule: none, cosine (warmup+decay)")
    parser.add_argument("--warmup-fraction", type=float, default=0.05, help="Fraction of steps for linear LR warmup (cosine only)")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum LR for cosine decay")

    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Architectures: {args.architectures}")
    print(f"Output: {out_dir}")

    all_metrics = {}
    for arch in args.architectures:
        metrics = train_one_arch(arch, args, device, out_dir)
        all_metrics[arch] = metrics

    # Save plots
    save_plots(all_metrics, out_dir, args.show_plot)

    # Final comparison
    print(f"\n{'='*60}")
    print("Final Comparison")
    print(f"{'='*60}")

    results = []
    for arch, m in all_metrics.items():
        if m["steps"]:
            results.append({
                "arch": arch,
                "seq_acc": m["seq_acc"][-1],
                "tok_acc": m["token_acc"][-1],
                "full_loss": m["full_loss"][-1],
            })
            print(f"  {arch}: seq_acc={m['seq_acc'][-1]:.4f}  tok_acc={m['token_acc'][-1]:.4f}  loss={m['full_loss'][-1]:.4f}")

    # Sort by comparison rule
    results.sort(key=lambda r: (-r["seq_acc"], -r["tok_acc"], r["full_loss"]))
    if results:
        print(f"\nWinner: {results[0]['arch']}")


if __name__ == "__main__":
    main()
