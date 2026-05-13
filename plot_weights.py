"""
plot_weights.py  —  Weight distribution analysis for a single model layer.

Plots four panels:
  1. Full weight histogram  — overall distribution shape and outlier tails
  2. P1-P99 body            — distribution without outliers, reveals true shape
  3. Per-block scale spread — variance between block-level scales
  4. Quantization error     — simulated int8 blockwise error distribution

Usage:
    python plot_weights.py model.layers.0.self_attn.q_proj
    python plot_weights.py model.layers.0.self_attn.q_proj --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoModelForCausalLM


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK = 64


# ── Layer navigation ──────────────────────────────────────────────────────────

def get_layer(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Navigate a model by dotted attribute path.
    Integer segments index into ModuleList / Sequential.
    e.g. 'model.layers.0.self_attn.q_proj'
    """
    obj = model
    for part in path.split("."):
        try:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        except (AttributeError, IndexError, TypeError) as e:
            raise AttributeError(
                f"Could not resolve '{part}' in path '{path}': {e}"
            ) from e
    return obj


# ── Quantization simulation ───────────────────────────────────────────────────

def simulate_int8_blockwise(weight: np.ndarray) -> np.ndarray:
    """
    Simulate blockwise int8 quantization on a 2-D [N, K] weight array.
    Returns the per-element absolute error |original - dequantized|.
    K is trimmed to the nearest multiple of BLOCK if necessary.
    """
    N, K = weight.shape
    K = (K // BLOCK) * BLOCK          # trim to block boundary
    w = weight[:, :K].reshape(N, K // BLOCK, BLOCK)

    scales = np.abs(w).max(axis=2, keepdims=True).clip(1e-8) / 127.0
    qw     = np.round(w / scales).clip(-127, 127)
    error  = np.abs(w - qw * scales)
    return error.flatten()


# ── Statistics ────────────────────────────────────────────────────────────────

def print_stats(w: np.ndarray, layer_path: str) -> None:
    p1, p25, p45, p50, p55, p75, p99 = np.percentile(w, [1, 25, 45, 50, 55, 75, 99])
    outlier_pct = (np.abs(w - w.mean()) > 3 * w.std()).mean() * 100
    print(f"\n{'─'*54}")
    print(f"  Layer    : {layer_path}")
    print(f"  Elements : {w.size:,}")
    print(f"  Mean     : {w.mean():.6f}")
    print(f"  Std      : {w.std():.6f}")
    print(f"  Min/Max  : {w.min():.6f} / {w.max():.6f}")
    print(f"  P1  /P99 : {p1:.6f} / {p99:.6f}")
    print(f"  P25 /P75 : {p25:.6f} / {p75:.6f}")
    print(f"  P45 /P55 : {p45:.6f} / {p55:.6f}")
    print(f"  P50      : {p50:.6f}")
    print(f"  Outliers (|w - mean| > 3sigma) : {outlier_pct:.2f}%")
    print(f"{'─'*54}")


def print_recommendation(w: np.ndarray, error: np.ndarray) -> None:
    outlier_pct = (np.abs(w - w.mean()) > 3 * w.std()).mean() * 100
    snr         = w.std() / (error.std() + 1e-10)
    mean_err    = error.mean()

    print("\n── Quantization Recommendation ──────────────────────")
    print(f"  Outlier ratio  : {outlier_pct:.2f}%   (good < 1%,  concern > 5%)")
    print(f"  SNR (std/noise): {snr:.1f}x          (good > 50x, concern < 20x)")
    print(f"  Mean abs error : {mean_err:.6f}")

    if outlier_pct < 1.0 and snr > 50:
        verdict = "GOOD  — int8 blockwise quantization should work well for this layer."
    elif outlier_pct < 3.0 and snr > 20:
        verdict = (
            "CAUTION — moderate outliers present. Expect minor quality loss.\n"
            "          Consider clipping weights at ±3sigma before quantizing."
        )
    else:
        verdict = (
            "WARNING — heavy outliers or low SNR. Options to consider:\n"
            "          1. Clip weights at ±3sigma before quantizing.\n"
            "          2. Reduce block size (e.g. 32) to tighten per-block scales.\n"
            "          3. Keep this layer in float16 and skip quantization."
        )
    print(f"\n  {verdict}")
    print("─────────────────────────────────────────────────────\n")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(
    w_flat      : np.ndarray,
    error_flat  : np.ndarray,
    block_scales: np.ndarray,
    layer_path  : str,
    weight_shape: tuple,
) -> str:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Weight Analysis: {layer_path}   shape={weight_shape}",
        fontsize=12,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    mean = w_flat.mean()
    std  = w_flat.std()

    # ── 1. Full weight distribution ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(w_flat, bins=300, color="steelblue", alpha=0.75, density=True)
    ax1.axvline(mean, color="red", lw=1.5, linestyle="--", label=f"mean={mean:.4f}")
    for sigma, label in [(1, "±1σ"), (2, "±2σ"), (3, "±3σ")]:
        ax1.axvline(mean + sigma * std, color="orange", lw=0.8,
                    linestyle=":", label=label if sigma == 1 else "")
        ax1.axvline(mean - sigma * std, color="orange", lw=0.8, linestyle=":")
    ax1.set_title("Full weight distribution")
    ax1.set_xlabel("Weight value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=7)

    # ── 2. Body: P1–P99 ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    p1, p99 = np.percentile(w_flat, [1, 99])
    body = w_flat[(w_flat >= p1) & (w_flat <= p99)]
    ax2.hist(body, bins=300, color="steelblue", alpha=0.75, density=True)
    ax2.axvline(mean, color="red", lw=1.5, linestyle="--", label=f"mean={mean:.4f}")
    ax2.set_title(f"Body: P1–P99  [{p1:.4f}, {p99:.4f}]\n"
                  f"(outliers beyond this range excluded)")
    ax2.set_xlabel("Weight value")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7)

    # ── 3. Per-block scale distribution ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(block_scales, bins=100, color="mediumseagreen", alpha=0.75)
    cv = block_scales.std() / (block_scales.mean() + 1e-10)   # coefficient of variation
    ax3.set_title(f"Per-block scale values  ({block_scales.size:,} blocks of {BLOCK})\n"
                  f"High spread → blocks have very different dynamic ranges")
    ax3.set_xlabel(f"Scale  (max|block| / 127)")
    ax3.set_ylabel("Block count")
    ax3.text(
        0.97, 0.95,
        f"mean = {block_scales.mean():.5f}\nstd  = {block_scales.std():.5f}\nCV   = {cv:.2f}",
        transform=ax3.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    # ── 4. Quantization error ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(error_flat, bins=200, color="tomato", alpha=0.75)
    ax4.set_title(f"Int8 blockwise quantization error  (block={BLOCK})\n"
                  f"Simulated: |original weight − dequantized weight|")
    ax4.set_xlabel("|original − dequantized|")
    ax4.set_ylabel("Count")
    ax4.text(
        0.97, 0.95,
        f"mean  = {error_flat.mean():.6f}\nmax   = {error_flat.max():.6f}\n"
        f"p99   = {np.percentile(error_flat, 99):.6f}",
        transform=ax4.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    safe_name = layer_path.replace(".", "_")
    out_path  = f"hist_{safe_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot weight histogram and quantization analysis for one layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python plot_weights.py model.layers.0.self_attn.q_proj",
    )
    parser.add_argument(
        "layer",
        help="Dotted path to the layer, e.g. model.layers.0.self_attn.q_proj",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # ── Get layer weights ─────────────────────────────────────────────────────
    try:
        layer = get_layer(model, args.layer)
    except AttributeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(layer, "weight"):
        print(
            f"ERROR: '{args.layer}' has no .weight attribute.\n"
            "Make sure the path points to a Linear (or similar) layer, not a block.",
            file=sys.stderr,
        )
        sys.exit(1)

    weight_tensor = layer.weight.data
    print(f"Layer : {args.layer}")
    print(f"Shape : {tuple(weight_tensor.shape)}")

    # Flatten to 2D [N, K] — same logic as quantize.py
    if weight_tensor.dim() > 2:
        weight_2d = weight_tensor.float().cpu().numpy().reshape(weight_tensor.shape[0], -1)
    else:
        weight_2d = weight_tensor.float().cpu().numpy()

    w_flat = weight_2d.flatten()

    # ── Simulate quantization ─────────────────────────────────────────────────
    error_flat = simulate_int8_blockwise(weight_2d)

    # Per-block scales for panel 3
    N, K = weight_2d.shape
    K_trim = (K // BLOCK) * BLOCK
    w_blocks     = weight_2d[:, :K_trim].reshape(N, K_trim // BLOCK, BLOCK)
    block_scales = np.abs(w_blocks).max(axis=2).flatten() / 127.0

    # ── Print stats and recommendation ───────────────────────────────────────
    print_stats(w_flat, args.layer)
    print_recommendation(w_flat, error_flat)

    # ── Save plot ─────────────────────────────────────────────────────────────
    out_path = plot(w_flat, error_flat, block_scales, args.layer, tuple(weight_tensor.shape))
    print(f"Saved : {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
