"""
LLaMA/Qwen  —  Mixed INT8/FP16 Block Quantization with Outlier Bitmask
=======================================================================
Within each block of BLOCK_SIZE values:
  - Outlier values (|x − μ| > OUTLIER_SIGMA · σ) are kept as float16.
  - The rest are quantised to int8 with one float16 scale per block.
  - A 64-bit bitmask records which positions are float16 (bit i = 1 → pos i
    is float16; LSB = position 0).

Storage layout per block (CUDA-kernel friendly, CSR-style):
  bitmask      : int64   — 1 value per block
  scale        : float16 — 1 value per block
  int8_data    : int8[]  — packed, position-ascending, contiguous across blocks
  fp16_data    : float16[] — same ordering convention
  int8_offsets : int32[] — CSR prefix-sum of int8 counts  (len = num_blocks+1)
  fp16_offsets : int32[] — CSR prefix-sum of float16 counts (len = num_blocks+1)

Only layers listed in LAYERS_FILE are quantised; every other layer is unchanged.
"""

import copy
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_SIZE    = 1024 # values per quantisation block  ← tweak here
OUTLIER_SIGMA = 3.0      # positions beyond this many σ are kept as float16 (mixed mode only)
MAX_NEW       = 50
LAYERS_FILE   = "layers_to_quantize.txt"

# Quantisation mode — choose one:
#   "mixed"    : outliers stored as float16 + bitmask per block (MixedQuantLinear)
#   "pure_int8": 1 float16 scale + 64 int8 values per block, no outlier tracking (PureInt8Linear)
QUANT_MODE       = "mixed"

# ── Inference speed-ups ───────────────────────────────────────────────────────
USE_COMPILE      = True      # Plan B: wrap hot paths with torch.compile
COMPILE_MODE     = "default" # "default" | "reduce-overhead" | "max-autotune"
USE_FUSED_MATMUL = True      # Plan C: fused block matmul for PureInt8Linear

# ── Companding (pure_int8 only) ───────────────────────────────────────────────
COMPANDING = "mu_law"   # "none" | "mu_law" | "a_law"
MU_LAW_MU  = 255.0    # μ parameter for mu-law
A_LAW_A    = 87.6     # A parameter for A-law


# ── Data structure ────────────────────────────────────────────────────────────
@dataclass
class MixedQuantWeight:
    """
    Mixed INT8/FP16 quantised representation of a single weight tensor.

    CUDA kernel contract
    --------------------
    * int8_data and fp16_data are 1-D contiguous tensors.
    * int8_offsets[b] and fp16_offsets[b] are the inclusive start indices in
      int8_data / fp16_data for block b; the exclusive end is offsets[b+1].
    * Within each block, values are stored in ascending position order so that
      scatter / gather reduces to a single indexed write per value.
    * Bit i of bitmask[b] (LSB = position 0) is 1 iff position i in block b
      is stored as float16.  bitmask is [num_blocks, W] where
      W = ceil(BLOCK_SIZE / 64); word w covers positions [w*64, (w+1)*64).
    """
    int8_data:      torch.Tensor    # [total_int8_count]          int8
    fp16_data:      torch.Tensor    # [total_fp16_count]          float16
    bitmask:        torch.Tensor    # [num_blocks, ceil(S/64)]    int64
    scales:         torch.Tensor    # [num_blocks]          float16
    int8_offsets:   torch.Tensor    # [num_blocks + 1]      int32
    fp16_offsets:   torch.Tensor    # [num_blocks + 1]      int32
    original_shape: tuple


# ── Quantisation ──────────────────────────────────────────────────────────────
def quantize_mixed(weight: torch.Tensor) -> MixedQuantWeight:
    """
    Quantise *weight* to mixed INT8/FP16 block format.

    All arithmetic is done in float32 for precision; outputs are int8 / float16.
    The operation is fully vectorised at the block level — no Python loops over
    individual blocks.
    """
    flat = weight.detach().float().reshape(-1)

    pad = (-flat.numel()) % BLOCK_SIZE
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])

    num_blocks = flat.numel() // BLOCK_SIZE
    blocks = flat.view(num_blocks, BLOCK_SIZE)          # [B, S]

    # ── Outlier detection ─────────────────────────────────────────────────────
    mu  = blocks.mean(dim=1, keepdim=True)              # [B, 1]
    sig = blocks.std(dim=1,  keepdim=True).clamp(min=1e-8)
    outlier_mask = (blocks - mu).abs() > OUTLIER_SIGMA * sig   # [B, S]  bool

    # ── Build bitmasks — ceil(BLOCK_SIZE/64) int64 words per block ───────────
    # Each word covers 64 positions; shifts stay within [0, 63] so no overflow.
    num_words = max(1, (BLOCK_SIZE + 63) // 64)
    bitmasks  = torch.zeros(num_blocks, num_words, dtype=torch.int64)
    for w in range(num_words):
        lo = w * 64
        hi = min(lo + 64, BLOCK_SIZE)
        local_pos        = torch.arange(hi - lo, dtype=torch.int64)
        word_vals        = outlier_mask[:, lo:hi].to(torch.int64) << local_pos
        bitmasks[:, w]   = word_vals.sum(dim=1)

    # ── Per-block scales (computed on non-outlier values only) ───────────────
    safe_blocks = blocks.clone()
    safe_blocks[outlier_mask] = 0.0
    abs_max = safe_blocks.abs().amax(dim=1).clamp(min=1e-8)     # [B]
    scales  = (abs_max / 127.0).to(torch.float16)               # [B]

    # ── Quantise everything to int8; outlier slots get 0 (ignored on decode) ─
    qblocks = (
        blocks / abs_max.unsqueeze(1)
    ).mul(127).round().clamp(-127, 127).to(torch.int8)          # [B, S]

    # ── Pack into contiguous 1-D arrays (boolean indexing preserves order) ───
    int8_data = qblocks[~outlier_mask]                          # [total_int8]   int8
    fp16_data = blocks[outlier_mask].to(torch.float16)          # [total_fp16]  float16

    int8_counts = (~outlier_mask).sum(dim=1).to(torch.int32)    # [B]
    fp16_counts =   outlier_mask.sum(dim=1).to(torch.int32)     # [B]

    int8_offsets = torch.zeros(num_blocks + 1, dtype=torch.int32)
    fp16_offsets = torch.zeros(num_blocks + 1, dtype=torch.int32)
    int8_offsets[1:] = int8_counts.cumsum(0)
    fp16_offsets[1:] = fp16_counts.cumsum(0)

    return MixedQuantWeight(
        int8_data     = int8_data,
        fp16_data     = fp16_data,
        bitmask       = bitmasks,
        scales        = scales,
        int8_offsets  = int8_offsets,
        fp16_offsets  = fp16_offsets,
        original_shape= tuple(weight.shape),
    )


# ── Dequantisation ────────────────────────────────────────────────────────────
def dequantize_mixed(mqw: MixedQuantWeight) -> torch.Tensor:
    """
    Reconstruct a float16 weight tensor from *mqw*.

    The three steps below map directly to a CUDA kernel:
      1. Expand each int64 bitmask to a per-position bool via vectorised shifts.
      2. Compute the flat 1-D scatter index for every value (block_start + pos).
      3. Scatter float16 outliers and dequantised int8 values in two writes.
    """
    num_blocks = mqw.bitmask.shape[0]
    num_words  = mqw.bitmask.shape[1]
    device     = mqw.int8_data.device

    flat = torch.zeros(num_blocks * BLOCK_SIZE, dtype=torch.float16, device=device)

    # ── Step 1 — expand multi-word bitmask → bool mask [B, S] ────────────────
    # Process each 64-bit word separately so shifts stay within [0, 63].
    parts = []
    for w in range(num_words):
        lo = w * 64
        hi = min(lo + 64, BLOCK_SIZE)
        local_pos = torch.arange(hi - lo, dtype=torch.int64, device=device)
        part = ((mqw.bitmask[:, w].unsqueeze(1) >> local_pos) & 1).bool()
        parts.append(part)
    fp16_mask = torch.cat(parts, dim=1)     # [B, S]
    int8_mask  = ~fp16_mask

    # ── Step 2 — flat scatter positions ──────────────────────────────────────
    pos_in_block = torch.arange(BLOCK_SIZE, device=device)
    block_starts = torch.arange(num_blocks, device=device).unsqueeze(1) * BLOCK_SIZE
    all_pos  = block_starts + pos_in_block  # [B, S]
    fp16_pos = all_pos[fp16_mask]   # [total_fp16]
    int8_pos = all_pos[int8_mask]   # [total_int8]

    # ── Step 3 — scatter ──────────────────────────────────────────────────────
    flat[fp16_pos] = mqw.fp16_data.to(device)

    block_idx      = int8_mask.nonzero(as_tuple=False)[:, 0]       # [total_int8]
    scales_per_val = mqw.scales[block_idx].to(device)              # [total_int8]
    flat[int8_pos] = mqw.int8_data.to(torch.float16).to(device) * scales_per_val

    # ── Trim padding, restore original shape ─────────────────────────────────
    original_numel = 1
    for s in mqw.original_shape:
        original_numel *= s
    return flat[:original_numel].reshape(mqw.original_shape)


# ── Module ────────────────────────────────────────────────────────────────────
class MixedQuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with mixed INT8/float16 weight storage.

    Forward pass
    ------------
    1. Dequantise all weights back to float16  (dequantize_mixed).
    2. Run the original F.linear with the reconstructed weight matrix.

    All six quantised tensors are registered as buffers so that .to(device)
    and state_dict() work transparently.
    """

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
            int8_data     = self.int8_data,
            fp16_data     = self.fp16_data,
            bitmask       = self.bitmask,
            scales        = self.scales,
            int8_offsets  = self.int8_offsets,
            fp16_offsets  = self.fp16_offsets,
            original_shape= self._weight_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantise weights to float16 then run original linear forward.
        weight = dequantize_mixed(self._mqw())
        bias   = self.bias.to(weight.dtype) if self.bias is not None else None
        return F.linear(x.to(weight.dtype), weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"block={BLOCK_SIZE}, sigma={OUTLIER_SIGMA}"
        )


# ── Companding helpers (pure_int8 only) ──────────────────────────────────────
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
    return x  # "none"


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
    return y  # "none"


# ── Pure INT8 block quantisation (no outlier tracking) ───────────────────────
def quantize_int8_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-block int8 quantisation with no outlier tracking.

    Each row is padded to a multiple of BLOCK_SIZE then sliced into blocks.
    One scale = max(|block|) / 127 is stored per block.

    Returns
    -------
    qweight : int8   [N, K_padded]          — quantised weights
    scales  : float16 [N, K_padded // BLOCK_SIZE] — per-block scales
    """
    if weight.dim() > 2:
        weight = weight.reshape(weight.shape[0], -1)

    N, K = weight.shape
    pad  = (-K) % BLOCK_SIZE
    if pad:
        weight = torch.cat([weight, weight.new_zeros(N, pad)], dim=1)
    K_pad = weight.shape[1]

    w      = weight.float().view(N, K_pad // BLOCK_SIZE, BLOCK_SIZE)   # [N, B, S]
    scales = w.abs().amax(dim=2).clamp(min=1e-8)                       # [N, B]  absmax
    w_norm = w / scales.unsqueeze(2)                                   # [N, B, S]  ∈ [−1, 1]
    w_comp = _compand(w_norm)                                          # same shape, companded
    qweight = w_comp.mul(127).round().clamp(-127, 127).to(torch.int8)
    return qweight.view(N, K_pad), scales.to(torch.float16)


def dequantize_int8_block(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple,
) -> torch.Tensor:
    """
    Reconstruct a float16 weight tensor from pure int8 block storage.
    """
    N, K_pad   = qweight.shape
    num_blocks = K_pad // BLOCK_SIZE
    w_comp = qweight.view(N, num_blocks, BLOCK_SIZE).to(torch.float16) / 127.0  # ∈ [−1, 1]
    w_norm = _expand(w_comp)                                                     # inverse compand
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
    Plan C — block-by-block fused dequant + matmul.

    Reshapes both x and the weight into [*, num_blocks, BLOCK_SIZE] and
    computes a single einsum, so TorchInductor can fuse the scale multiply
    directly into the GEMM without ever allocating a full [N, K_pad] fp16
    weight tensor.

    qweight : [N, K_pad]               int8
    scales  : [N, K_pad // BLOCK_SIZE] float16
    x       : [..., K_input]           any float dtype
    """
    N, K_pad   = qweight.shape
    num_blocks = K_pad // BLOCK_SIZE

    x_shape = x.shape
    x_flat  = x.reshape(-1, x_shape[-1]).to(torch.float16)  # [B, K_input]
    batch   = x_flat.shape[0]

    k_in = x_flat.shape[1]
    if k_in < K_pad:
        x_flat = F.pad(x_flat, (0, K_pad - k_in))
    elif k_in > K_pad:
        x_flat = x_flat[:, :K_pad]

    x_blocked = x_flat.view(batch, num_blocks, BLOCK_SIZE)          # [B, nb, S]
    w_blocked = (
        qweight.view(N, num_blocks, BLOCK_SIZE).to(torch.float16)
        * scales.unsqueeze(2)
    )                                                                # [N, nb, S]

    # out[i, n] = Σ_b Σ_k  x[i,b,k] * w[n,b,k]
    # Cast to float32 for accumulation — float16 can overflow for long sequences.
    out = torch.einsum("ibk,nbk->in", x_blocked.float(), w_blocked.float()).to(torch.float16)

    out = out.reshape(*x_shape[:-1], N)
    if bias is not None:
        out = out + bias.to(torch.float16)
    return out


class PureInt8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear — pure int8 block storage, no bitmask.

    Storage per block of BLOCK_SIZE values: 1 float16 scale + BLOCK_SIZE int8 weights.
    No outlier handling; outlier values are simply clipped to ±127 via the scale.

    Forward pass
    ------------
    1. Dequantise all weights to float16 (dequantize_int8_block).
    2. Run the original F.linear with the reconstructed weight matrix.
    """

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


# ── Layer selection from file ─────────────────────────────────────────────────
def load_layer_names(path: str) -> list[str]:
    """
    Read dot-separated layer paths from *path*, one per line.
    Lines starting with '#' and blank lines are ignored.
    """
    names = []
    with open(path) as fh:
        for line in fh:
            name = line.split("#")[0].strip()
            if name:
                names.append(name)
    return names


def quantize_selected_layers(model: nn.Module, layers_file: str) -> nn.Module:
    """
    Replace every nn.Linear listed in *layers_file* with a MixedQuantLinear.
    Layers not in the file are left untouched.
    """
    names = load_layer_names(layers_file)
    print(f"  Layers requested : {len(names)}")

    for full_name in names:
        parts       = full_name.split(".")
        parent_path = ".".join(parts[:-1])
        attr        = parts[-1]

        try:
            parent = model.get_submodule(parent_path) if parent_path else model
            child  = getattr(parent, attr)
        except (AttributeError, KeyError):
            print(f"  [skip] {full_name}  — not found in model")
            continue

        if not isinstance(child, nn.Linear):
            print(f"  [skip] {full_name}  — expected nn.Linear, got {type(child).__name__}")
            continue

        cls = MixedQuantLinear if QUANT_MODE == "mixed" else PureInt8Linear
        setattr(parent, attr, cls(child))
        print(f"  [done] {full_name}  ({QUANT_MODE})")

    return model


# ── Utilities ─────────────────────────────────────────────────────────────────
def model_size_mb(model: nn.Module) -> float:
    params  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (params + buffers) / 1024 ** 2


def count_layers(model: nn.Module) -> tuple[int, int]:
    quant = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, MixedQuantLinear, PureInt8Linear)):
            total += 1
            if isinstance(m, (MixedQuantLinear, PureInt8Linear)):
                quant += 1
    return quant, total


# ── Plan B — torch.compile hot paths ─────────────────────────────────────────
# Replaces the module-level names so every forward call uses compiled versions.
# First call per layer triggers JIT compilation (warm-up); subsequent calls are
# faster. Remove or set USE_COMPILE = False to disable.
if USE_COMPILE:
    _c = lambda f: torch.compile(f, mode=COMPILE_MODE)
    dequantize_mixed      = _c(dequantize_mixed)
    dequantize_int8_block = _c(dequantize_int8_block)
    # dynamic=True: one compiled graph handles variable sequence lengths without recompiling each step
    _int8_fused_matmul    = torch.compile(_int8_fused_matmul, mode=COMPILE_MODE, dynamic=True)

def _first_non_meta_param_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        if getattr(p, "device", None) is not None and str(p.device) != "meta":
            return p.device
    return torch.device("cpu")

def _inputs_for_model(model: torch.nn.Module, tokenizer, text: str):
    batch = tokenizer(text, return_tensors="pt")
    device = _first_non_meta_param_device(model)
    return batch.to(device)

def compute_loss(model, tokenizer,text: str):
    batch = _inputs_for_model(model,tokenizer, text)
    with torch.no_grad():
        outputs = model(**batch, labels=batch["input_ids"])
    return outputs.loss.item()

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    texts = [
    "Digital signal processing improves efficiency.",
    "Quantization reduces model size."
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    size_fp16 = model_size_mb(model)
    print(f"  Float16 size      : {size_fp16:,.1f} MB")

    print(f"\nQuantizing layers from '{LAYERS_FILE}' ...")
    t0 = time.time()
    with torch.no_grad():
        quantize_selected_layers(model, LAYERS_FILE)
    dt = time.time() - t0

    size_mixed   = model_size_mb(model)
    quant, total = count_layers(model)
    print(f"  Done in           : {dt:.1f} s")
    print(f"  Mixed-quant size  : {size_mixed:,.1f} MB  (was {size_fp16:,.1f} MB)")
    print(f"  Layers quantised  : {quant} / {total}")

    model = model.to(device)

    prompt = "The meaning of life is"
    print(f"\nPrompt : {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    dt = time.time() - t0

    n_new   = output_ids.shape[1] - inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output : {decoded!r}")
    print(f"  {n_new} tokens in {dt:.2f} s  ({n_new / dt:.1f} tok/s)\n")
    loss_quant = compute_loss(model,tokenizer, texts[0])
    print(f"  Quant loss        : {loss_quant:.4f}")



if __name__ == "__main__":
    main()
