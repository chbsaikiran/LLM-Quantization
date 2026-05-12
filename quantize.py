"""
LLaMA 3B  —  Int8 Blockwise Quantization
=========================================
Every nn.Linear (except lm_head) is quantized to int8 with one float16
scale per 64-weight block.  At inference time we dequantize one block at
a time and accumulate partial matmuls — K//64 Python iterations per layer
instead of the naive N * K//64.

Requirements:
    pip install torch transformers accelerate

HuggingFace access:
    The model is gated. Accept the licence at
    https://huggingface.co/meta-llama/Llama-3.2-3B
    then run:  huggingface-cli login
"""

import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK     = 64          # weights per quantization block
MAX_NEW   = 50          # tokens to generate in the demo


# ── Core quantization function ────────────────────────────────────────────────

def quantize_int8_blockwise(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-block int8 quantization.

    Each row is sliced into non-overlapping blocks of BLOCK elements.
    One scale = max(|block|) / 127 is stored per block.

    Args:
        weight : float tensor  [N, K]   — original linear weight

    Returns:
        qweight : int8   tensor  [N, K]          — quantized weights
        scales  : float16 tensor [N, K // BLOCK]  — per-block scales
    """
    # Flatten any extra dims into K so quantization always operates on [N, K].
    # e.g. a Conv weight [C_out, C_in, kH, kW] becomes [C_out, C_in*kH*kW].
    if weight.dim() > 2:
        weight = weight.view(weight.shape[0], -1)

    N, K = weight.shape
    assert K % BLOCK == 0, (
        f"Weight K-dim ({K}) must be divisible by BLOCK ({BLOCK}). "
        "Pad the linear layer before quantizing if needed."
    )

    # Expose blocks: [N, K] -> [N, num_blocks, BLOCK]
    w = weight.float().view(N, K // BLOCK, BLOCK)

    # Per-block absolute max, clamped to avoid division by zero: [N, num_blocks]
    scales = w.abs().amax(dim=2).clamp(min=1e-8)

    # Divide by scale, multiply by 127, round, clamp to [-127, 127]
    qweight = (
        (w / scales.unsqueeze(2))
        .mul(127)
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
    )

    return qweight.view(N, K), scales.to(torch.float16)


# ── QuantLinear ───────────────────────────────────────────────────────────────

class QuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear.

    Weights are stored in int8 with per-block float16 scales.
    Forward dequantizes one 64-column block at a time so that each
    iteration is a proper [B, 64] @ [64, N] matmul — eliminating the
    inner loop over N from the naive implementation.

    Loop count per forward call: K // 64   (e.g. 64 for a 4096-wide layer)
    vs naive:                    N * K // 64  (e.g. 262 144 for same layer)
    """

    BLOCK = 64

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()

        qweight, scales = quantize_int8_blockwise(linear.weight.data)

        self.register_buffer("qweight", qweight)   # [N, K]          int8
        self.register_buffer("scales",  scales)    # [N, K // BLOCK]  float16

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

        self.in_features  = linear.in_features
        self.out_features = linear.out_features

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape

        # LLaMA passes [batch, seq_len, hidden] — flatten to [B*S, K]
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])

        B, K = x.shape
        N    = self.qweight.shape[0]

        out = torch.zeros(B, N, device=x.device, dtype=x.dtype)

        for b in range(K // self.BLOCK):
            start = b * self.BLOCK
            end   = start + self.BLOCK

            x_block = x[:, start:end]                           # [B,  64]
            q_block = self.qweight[:, start:end]                # [N,  64]  int8
            scale   = self.scales[:, b].to(x.dtype)            # [N]       float16

            # Dequantize this block only:  [N, 64]
            w_block = q_block.to(x.dtype) * scale.unsqueeze(1)

            # Partial matmul and accumulate:  [B, 64] @ [64, N] → [B, N]
            out += x_block @ w_block.T

        if self.bias is not None:
            out += self.bias

        # Restore original leading dims for 3-D inputs
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], N)

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, block={self.BLOCK}"
        )


# ── Model surgery ─────────────────────────────────────────────────────────────

def replace_linears(
    model : nn.Module,
    skip  : set[str] | None = None,
) -> nn.Module:
    """
    Walk the module tree and swap every nn.Linear for a QuantLinear.

    Args:
        model : the model to patch in-place
        skip  : attribute names to leave as nn.Linear.
                Defaults to {"lm_head"} — the output projection maps
                hidden → vocab and is quality-sensitive enough to skip.
    """
    if skip is None:
        skip = {"lm_head"}

    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name not in skip:
            setattr(model, name, QuantLinear(child))
        else:
            replace_linears(child, skip)   # recurse into sub-modules

    return model


# ── Utilities ─────────────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    """Total parameter + buffer storage in megabytes."""
    params  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (params + buffers) / (1024 ** 2)


def count_layers(model: nn.Module) -> tuple[int, int]:
    """Return (num_quantized_linears, num_total_linears)."""
    quant = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, QuantLinear)):
            total += 1
            if isinstance(m, QuantLinear):
                quant += 1
    return quant, total


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    # ── 1. Load pretrained model ──────────────────────────────────────────────
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,   # stream weights in to avoid 2× peak RAM
    )
    model.eval()

    size_fp16 = model_size_mb(model)
    print(f"  Float16 size  : {size_fp16:,.0f} MB")

    # ── 2. Quantize in-place ──────────────────────────────────────────────────
    print(f"\nQuantizing (int8, block={BLOCK}) — skipping lm_head ...")
    t0 = time.time()
    with torch.no_grad():
        replace_linears(model)                     # skip={"lm_head"} default
    dt = time.time() - t0

    size_int8      = model_size_mb(model)
    quant, total   = count_layers(model)
    reduction      = size_fp16 / size_int8

    print(f"  Done in       : {dt:.1f}s")
    print(f"  Int8 size     : {size_int8:,.0f} MB  ({reduction:.1f}× smaller)")
    print(f"  Layers quant  : {quant} / {total}")

    # ── 3. Move to inference device ───────────────────────────────────────────
    model = model.to(device)

    # ── 4. Inference demo ─────────────────────────────────────────────────────
    prompt = "The meaning of life is"
    print(f"\nPrompt : {prompt!r}")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
        )
    dt = time.time() - t0

    n_new   = output_ids.shape[1] - inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Output : {decoded!r}")
    print(f"  {n_new} tokens in {dt:.2f}s  ({n_new / dt:.1f} tok/s)\n")


if __name__ == "__main__":
    main()
