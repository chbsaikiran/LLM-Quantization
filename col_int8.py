"""
Column-based outlier isolation quantisation (col_int8 mode).

Columns whose per-column absmax exceeds  mean + COL_OUTLIER_SIGMA × std  are
stored verbatim as float16.  All other columns are block-quantised to int8
(same scheme as pure_int8, with optional companding).

Forward pass — two standard GEMMs, no scatter, no bitmask:
    out = x[..., normal_cols]  @ W_int8_dequant.T   # INT8 block GEMM
        + x[..., outlier_cols] @ W_fp16.T             # FP16 GEMM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pure_int8
from pure_int8 import _expand, quantize_int8_block
from config import BLOCK_SIZE, COL_OUTLIER_SIGMA, USE_FUSED_MATMUL, USE_COMPILE, COMPILE_MODE


def _col_int8_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """
    Fused dequant + matmul for the non-outlier (INT8) columns.
    Correct dequant: int8 → /127 → inverse compand → × absmax scale.
    Accumulates in float32 to prevent fp16 overflow.
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
    w_comp    = qweight.view(N, num_blocks, BLOCK_SIZE).to(torch.float16) / 127.0
    w_norm    = _expand(w_comp)
    w_blocked = w_norm * scales.unsqueeze(2)
    out = torch.einsum("ibk,nbk->in", x_blocked.float(), w_blocked.float()).to(torch.float16)
    return out.reshape(*x_shape[:-1], N)


class ColOutlierLinear(nn.Module):
    """Drop-in nn.Linear replacement using column-based outlier isolation."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.data.float()   # [N, K]
        N, K   = weight.shape

        col_absmax   = weight.abs().amax(dim=0)
        threshold    = col_absmax.mean() + COL_OUTLIER_SIGMA * col_absmax.std().clamp(min=1e-8)
        outlier_mask = col_absmax > threshold

        normal_idx  = (~outlier_mask).nonzero(as_tuple=False).squeeze(1)
        outlier_idx =   outlier_mask.nonzero(as_tuple=False).squeeze(1)

        self.register_buffer("normal_col_idx",  normal_idx)
        self.register_buffer("outlier_col_idx", outlier_idx)

        if outlier_idx.numel() > 0:
            self.register_buffer("W_fp16", linear.weight.data[:, outlier_idx].to(torch.float16))
        else:
            self.W_fp16 = None

        if normal_idx.numel() > 0:
            W_int8, scales = quantize_int8_block(linear.weight.data[:, normal_idx])
            self.register_buffer("W_int8",        W_int8)
            self.register_buffer("W_int8_scales", scales)
        else:
            self.W_int8        = None
            self.W_int8_scales = None

        self._normal_K     = int(normal_idx.numel())
        self._weight_shape = (N, K)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

        self.in_features  = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(torch.float16)
        out    = None

        if self.W_int8 is not None:
            x_normal = x_fp16[..., self.normal_col_idx]
            if USE_FUSED_MATMUL:
                out = _col_int8_matmul(x_normal, self.W_int8, self.W_int8_scales)
            else:
                W   = pure_int8.dequantize_int8_block(
                    self.W_int8, self.W_int8_scales,
                    (self.out_features, self._normal_K),
                )
                out = F.linear(x_normal, W)

        if self.W_fp16 is not None:
            x_outlier = x_fp16[..., self.outlier_col_idx]
            fp16_out  = F.linear(x_outlier, self.W_fp16)
            out       = fp16_out if out is None else out + fp16_out

        if out is None:
            out = torch.zeros(*x.shape[:-1], self.out_features,
                              dtype=torch.float16, device=x.device)

        if self.bias is not None:
            out = out + self.bias.to(torch.float16)
        return out

    def extra_repr(self) -> str:
        n_out = self.outlier_col_idx.numel()
        n_tot = self._weight_shape[1]
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"outlier_cols={n_out}/{n_tot} ({100*n_out/max(n_tot,1):.1f}%), "
            f"block={BLOCK_SIZE}, sigma={COL_OUTLIER_SIGMA}"
        )


# ── Plan B: compile hot path ──────────────────────────────────────────────────
# dynamic=True: one compiled graph handles variable sequence lengths.
if USE_COMPILE:
    _col_int8_matmul = torch.compile(_col_int8_matmul, mode=COMPILE_MODE, dynamic=True)
