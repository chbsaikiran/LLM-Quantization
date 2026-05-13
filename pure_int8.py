"""
Pure INT8 block quantisation with optional mu-law / A-law companding.

Each row is padded to a multiple of BLOCK_SIZE then divided into blocks.
One absmax scale per block; no outlier tracking.

Companding pipeline (encode):
    absmax scale → normalise to [−1,1] → compand → quantise to int8

Inverse (decode):
    int8 / 127 → inverse compand → × scale → float16
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    BLOCK_SIZE, COMPANDING, MU_LAW_MU, A_LAW_A,
    USE_FUSED_MATMUL, USE_COMPILE, COMPILE_MODE,
)


# ── Companding helpers ────────────────────────────────────────────────────────

def _compand(x: torch.Tensor) -> torch.Tensor:
    """Map x ∈ [−1, 1] through the selected companding curve."""
    if COMPANDING == "mu_law":
        return x.sign() * (1.0 + MU_LAW_MU * x.abs()).log() / math.log(1.0 + MU_LAW_MU)
    if COMPANDING == "a_law":
        a     = A_LAW_A
        denom = 1.0 + math.log(a)
        abs_x = x.abs()
        return x.sign() * torch.where(
            abs_x < 1.0 / a,
            a * abs_x / denom,
            (1.0 + (a * abs_x).clamp(min=1e-8).log()) / denom,
        )
    return x


def _expand(y: torch.Tensor) -> torch.Tensor:
    """Inverse of _compand; map y ∈ [−1, 1] back to the linear domain."""
    if COMPANDING == "mu_law":
        return y.sign() * ((1.0 + MU_LAW_MU) ** y.abs() - 1.0) / MU_LAW_MU
    if COMPANDING == "a_law":
        a         = A_LAW_A
        denom     = 1.0 + math.log(a)
        abs_y     = y.abs()
        threshold = 1.0 / denom
        return y.sign() * torch.where(
            abs_y < threshold,
            abs_y * denom / a,
            (abs_y * denom - 1.0).exp() / a,
        )
    return y


# ── Quantisation / dequantisation ─────────────────────────────────────────────

def quantize_int8_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (qweight [N, K_pad] int8, scales [N, num_blocks] float16).
    """
    if weight.dim() > 2:
        weight = weight.reshape(weight.shape[0], -1)
    N, K = weight.shape
    pad  = (-K) % BLOCK_SIZE
    if pad:
        weight = torch.cat([weight, weight.new_zeros(N, pad)], dim=1)
    K_pad  = weight.shape[1]
    w      = weight.float().view(N, K_pad // BLOCK_SIZE, BLOCK_SIZE)
    scales = w.abs().amax(dim=2).clamp(min=1e-8)
    w_norm = w / scales.unsqueeze(2)
    w_comp = _compand(w_norm)
    qweight = w_comp.mul(127).round().clamp(-127, 127).to(torch.int8)
    return qweight.view(N, K_pad), scales.to(torch.float16)


def dequantize_int8_block(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple,
) -> torch.Tensor:
    N, K_pad   = qweight.shape
    num_blocks = K_pad // BLOCK_SIZE
    w_comp = qweight.view(N, num_blocks, BLOCK_SIZE).to(torch.float16) / 127.0
    w_norm = _expand(w_comp)
    w_fp16 = (w_norm * scales.unsqueeze(2)).view(N, K_pad)
    original_numel = 1
    for s in original_shape:
        original_numel *= s
    return w_fp16.reshape(N, -1)[:, :original_numel // N].reshape(original_shape)


def _int8_fused_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias,
) -> torch.Tensor:
    """
    Plan C — block-by-block fused dequant + matmul via einsum.
    Accumulates in float32 to prevent fp16 overflow on large activations.
    """
    N, K_pad   = qweight.shape
    num_blocks = K_pad // BLOCK_SIZE
    x_shape    = x.shape
    x_flat     = x.reshape(-1, x_shape[-1]).to(torch.float16)
    batch      = x_flat.shape[0]
    k_in       = x_flat.shape[1]
    if k_in < K_pad:
        x_flat = F.pad(x_flat, (0, K_pad - k_in))
    elif k_in > K_pad:
        x_flat = x_flat[:, :K_pad]
    x_blocked = x_flat.view(batch, num_blocks, BLOCK_SIZE)
    w_blocked = (
        qweight.view(N, num_blocks, BLOCK_SIZE).to(torch.float16) * scales.unsqueeze(2)
    )
    out = torch.einsum("ibk,nbk->in", x_blocked.float(), w_blocked.float()).to(torch.float16)
    out = out.reshape(*x_shape[:-1], N)
    if bias is not None:
        out = out + bias.to(torch.float16)
    return out


class PureInt8Linear(nn.Module):
    """Drop-in nn.Linear replacement using pure int8 block storage."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        qweight, scales = quantize_int8_block(linear.weight.data)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales",  scales)
        self._weight_shape = tuple(linear.weight.data.shape)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None
        self.in_features  = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if USE_FUSED_MATMUL:
            bias = self.bias.to(torch.float16) if self.bias is not None else None
            return _int8_fused_matmul(x, self.qweight, self.scales, bias)
        weight = dequantize_int8_block(self.qweight, self.scales, self._weight_shape)
        bias   = self.bias.to(weight.dtype) if self.bias is not None else None
        return F.linear(x.to(weight.dtype), weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, block={BLOCK_SIZE}"
        )


# ── Plan B: compile hot paths ─────────────────────────────────────────────────
# Replaces module-level names so forward calls pick up compiled versions.
if USE_COMPILE:
    dequantize_int8_block = torch.compile(dequantize_int8_block, mode=COMPILE_MODE)
    _int8_fused_matmul    = torch.compile(_int8_fused_matmul, mode=COMPILE_MODE, dynamic=True)
