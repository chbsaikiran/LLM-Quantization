"""
Mixed INT8/FP16 block quantisation with per-block outlier bitmask.

Within each block of BLOCK_SIZE values:
  - Outlier values (|x − μ| > OUTLIER_SIGMA · σ) are kept as float16.
  - The rest are quantised to int8 with one float16 scale per block.
  - A multi-word int64 bitmask records which positions are float16.

Storage layout (CSR-style, CUDA-kernel friendly):
  bitmask      : int64   [num_blocks, ceil(BLOCK_SIZE/64)]
  scales       : float16 [num_blocks]
  int8_data    : int8[]  packed, position-ascending, contiguous across blocks
  fp16_data    : float16[] same ordering
  int8_offsets : int32[] prefix-sum of int8 counts  (len = num_blocks+1)
  fp16_offsets : int32[] prefix-sum of float16 counts (len = num_blocks+1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from config import BLOCK_SIZE, OUTLIER_SIGMA, USE_COMPILE, COMPILE_MODE


@dataclass
class MixedQuantWeight:
    int8_data:      torch.Tensor    # [total_int8_count]          int8
    fp16_data:      torch.Tensor    # [total_fp16_count]          float16
    bitmask:        torch.Tensor    # [num_blocks, ceil(S/64)]    int64
    scales:         torch.Tensor    # [num_blocks]                float16
    int8_offsets:   torch.Tensor    # [num_blocks + 1]            int32
    fp16_offsets:   torch.Tensor    # [num_blocks + 1]            int32
    original_shape: tuple


def quantize_mixed(weight: torch.Tensor) -> MixedQuantWeight:
    flat = weight.detach().float().reshape(-1)

    pad = (-flat.numel()) % BLOCK_SIZE
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])

    num_blocks = flat.numel() // BLOCK_SIZE
    blocks = flat.view(num_blocks, BLOCK_SIZE)

    mu  = blocks.mean(dim=1, keepdim=True)
    sig = blocks.std(dim=1, keepdim=True).clamp(min=1e-8)
    outlier_mask = (blocks - mu).abs() > OUTLIER_SIGMA * sig   # [B, S] bool

    num_words = max(1, (BLOCK_SIZE + 63) // 64)
    bitmasks  = torch.zeros(num_blocks, num_words, dtype=torch.int64)
    for w in range(num_words):
        lo = w * 64
        hi = min(lo + 64, BLOCK_SIZE)
        local_pos      = torch.arange(hi - lo, dtype=torch.int64)
        word_vals      = outlier_mask[:, lo:hi].to(torch.int64) << local_pos
        bitmasks[:, w] = word_vals.sum(dim=1)

    safe_blocks = blocks.clone()
    safe_blocks[outlier_mask] = 0.0
    abs_max = safe_blocks.abs().amax(dim=1).clamp(min=1e-8)
    scales  = (abs_max / 127.0).to(torch.float16)

    qblocks  = (blocks / abs_max.unsqueeze(1)).mul(127).round().clamp(-127, 127).to(torch.int8)

    int8_data = qblocks[~outlier_mask]
    fp16_data = blocks[outlier_mask].to(torch.float16)

    int8_counts  = (~outlier_mask).sum(dim=1).to(torch.int32)
    fp16_counts  =   outlier_mask.sum(dim=1).to(torch.int32)
    int8_offsets = torch.zeros(num_blocks + 1, dtype=torch.int32)
    fp16_offsets = torch.zeros(num_blocks + 1, dtype=torch.int32)
    int8_offsets[1:] = int8_counts.cumsum(0)
    fp16_offsets[1:] = fp16_counts.cumsum(0)

    return MixedQuantWeight(
        int8_data      = int8_data,
        fp16_data      = fp16_data,
        bitmask        = bitmasks,
        scales         = scales,
        int8_offsets   = int8_offsets,
        fp16_offsets   = fp16_offsets,
        original_shape = tuple(weight.shape),
    )


def dequantize_mixed(mqw: MixedQuantWeight) -> torch.Tensor:
    num_blocks = mqw.bitmask.shape[0]
    num_words  = mqw.bitmask.shape[1]
    device     = mqw.int8_data.device

    flat = torch.zeros(num_blocks * BLOCK_SIZE, dtype=torch.float16, device=device)

    parts = []
    for w in range(num_words):
        lo = w * 64
        hi = min(lo + 64, BLOCK_SIZE)
        local_pos = torch.arange(hi - lo, dtype=torch.int64, device=device)
        part = ((mqw.bitmask[:, w].unsqueeze(1) >> local_pos) & 1).bool()
        parts.append(part)
    fp16_mask = torch.cat(parts, dim=1)
    int8_mask  = ~fp16_mask

    pos_in_block = torch.arange(BLOCK_SIZE, device=device)
    block_starts = torch.arange(num_blocks, device=device).unsqueeze(1) * BLOCK_SIZE
    all_pos  = block_starts + pos_in_block
    fp16_pos = all_pos[fp16_mask]
    int8_pos = all_pos[int8_mask]

    flat[fp16_pos] = mqw.fp16_data.to(device)

    block_idx      = int8_mask.nonzero(as_tuple=False)[:, 0]
    scales_per_val = mqw.scales[block_idx].to(device)
    flat[int8_pos] = mqw.int8_data.to(torch.float16).to(device) * scales_per_val

    original_numel = 1
    for s in mqw.original_shape:
        original_numel *= s
    return flat[:original_numel].reshape(mqw.original_shape)


class MixedQuantLinear(nn.Module):
    """Drop-in nn.Linear replacement using mixed INT8/float16 block storage."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        mqw = quantize_mixed(linear.weight.data)
        self.register_buffer("int8_data",    mqw.int8_data)
        self.register_buffer("fp16_data",    mqw.fp16_data)
        self.register_buffer("bitmask",      mqw.bitmask)
        self.register_buffer("scales",       mqw.scales)
        self.register_buffer("int8_offsets", mqw.int8_offsets)
        self.register_buffer("fp16_offsets", mqw.fp16_offsets)
        self._weight_shape = tuple(linear.weight.data.shape)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None
        self.in_features  = linear.in_features
        self.out_features = linear.out_features

    def _mqw(self) -> MixedQuantWeight:
        return MixedQuantWeight(
            int8_data      = self.int8_data,
            fp16_data      = self.fp16_data,
            bitmask        = self.bitmask,
            scales         = self.scales,
            int8_offsets   = self.int8_offsets,
            fp16_offsets   = self.fp16_offsets,
            original_shape = self._weight_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_mixed(self._mqw())
        bias   = self.bias.to(weight.dtype) if self.bias is not None else None
        return F.linear(x.to(weight.dtype), weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, block={BLOCK_SIZE}, sigma={OUTLIER_SIGMA}"
        )


# ── Plan B: compile hot path ──────────────────────────────────────────────────
# Replaces module-level name so MixedQuantLinear.forward picks up compiled version.
if USE_COMPILE:
    dequantize_mixed = torch.compile(dequantize_mixed, mode=COMPILE_MODE)
