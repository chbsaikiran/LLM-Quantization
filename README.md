# LLM Block Quantization from Scratch

Post-training quantization for Qwen2.5-1.5B-Instruct (and any compatible HuggingFace causal LM) implemented entirely from scratch in PyTorch — no bitsandbytes, no GPTQ, no external quantization library.

Two quantization schemes are provided, selectable via a single global flag:

| Mode | Storage per block | Outlier handling |
|---|---|---|
| `mixed` | INT8 values + FP16 outliers + bitmask | Yes — outliers kept in FP16 |
| `pure_int8` | INT8 values only | No — clipped by scale |

Both modes support optional **companding** (mu-law / A-law) and two inference speed-ups: **torch.compile** and a **fused block matmul**.

---

## Requirements

```
torch >= 2.2.0
transformers >= 4.40.0
accelerate >= 0.29.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Running

```bash
python quantize.py
```

All behaviour is controlled by globals at the top of `quantize.py` — no CLI flags needed.

---

## Configuration Reference

```python
# ── Core ──────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"   # HuggingFace model ID
BLOCK_SIZE    = 128        # values per quantisation block (tune this)
OUTLIER_SIGMA = 3.0        # outlier threshold in σ units  (mixed mode only)
MAX_NEW       = 50         # tokens to generate in the benchmark
LAYERS_FILE   = "layers_to_quantize.txt"        # which layers to quantize

# ── Quantisation mode ─────────────────────────────────────────────────────────
QUANT_MODE    = "mixed"    # "mixed"  → MixedQuantLinear
               # "pure_int8" → PureInt8Linear

# ── Inference speed-ups ───────────────────────────────────────────────────────
USE_COMPILE      = True    # wrap dequant/forward with torch.compile  (Plan B)
COMPILE_MODE     = "default"   # "default" | "reduce-overhead" | "max-autotune"
USE_FUSED_MATMUL = True    # block-by-block fused matmul, no full weight alloc (Plan C)
                            # only used when QUANT_MODE = "pure_int8"

# ── Companding (pure_int8 only) ───────────────────────────────────────────────
COMPANDING = "none"        # "none" | "mu_law" | "a_law"
MU_LAW_MU  = 255.0         # μ for mu-law
A_LAW_A    = 87.6          # A for A-law
```

### Quick mode recipes

```python
# Maximum quality, slowest inference
QUANT_MODE = "mixed";  BLOCK_SIZE = 64;  OUTLIER_SIGMA = 2.5

# Fastest inference, slightly lower quality
QUANT_MODE = "pure_int8";  USE_FUSED_MATMUL = True;  USE_COMPILE = True

# Pure INT8 with mu-law companding (better small-value fidelity)
QUANT_MODE = "pure_int8";  COMPANDING = "mu_law"

# Disable all speed-up tricks for debugging
USE_COMPILE = False;  USE_FUSED_MATMUL = False
```

---

## Layer Selection File (`layers_to_quantize.txt`)

One dot-separated module path per line. Lines starting with `#` and blank lines are ignored.

```
# Only quantize the first two transformer layers
model.layers.0.self_attn.q_proj
model.layers.0.self_attn.k_proj
model.layers.1.mlp.gate_proj
```

The provided `layers_to_quantize.txt` quantizes all 196 linear projections in Qwen2.5-1.5B (28 layers × 7 projections: q/k/v/o\_proj, gate/up/down\_proj). `lm_head` is intentionally excluded because it maps hidden states directly to vocabulary logits and is quality-sensitive.

---

## Mode 1 — Mixed INT8 / FP16 (`QUANT_MODE = "mixed"`)

### What it does

Each weight tensor is split into flat blocks of `BLOCK_SIZE` values. Within every block, values that deviate more than `OUTLIER_SIGMA` standard deviations from the block mean are classified as **outliers** and stored at full FP16 precision. The remaining values are quantized to INT8. A bitmask records which positions are FP16.

This mirrors the core idea behind bitsandbytes LLM.int8(), but implemented block-wise from scratch.

---

### Quantization Block Diagram

```
  Weight tensor  [N, K]  (fp32 working copy)
         │
         ▼
  ┌─────────────────────────────────────┐
  │  Flatten → 1-D  [N × K]            │
  │  Pad to multiple of BLOCK_SIZE      │
  │  Reshape → [num_blocks, BLOCK_SIZE] │
  └──────────────┬──────────────────────┘
                 │
         ┌───────▼────────┐
         │ Per-block stats │
         │  μ  = mean(block)│
         │  σ  = std(block) │
         └───────┬─────────┘
                 │
         ┌───────▼──────────────────────┐
         │  Outlier mask                 │
         │  is_outlier[i] =              │
         │    |block[i] − μ| > σ_thresh │
         │  where σ_thresh = OUTLIER_SIGMA × σ │
         └──────┬──────────┬────────────┘
                │          │
         ┌──────▼──┐  ┌────▼──────────────────┐
         │ Outlier  │  │ Normal values          │
         │ values   │  │                        │
         │          │  │ absmax = max(|normal|) │
         │ Kept as  │  │ scale  = absmax / 127  │
         │ FP16,    │  │                        │
         │ stored   │  │ q = round(v / absmax   │
         │ verbatim │  │         × 127)         │
         │          │  │ clamp to [−127, 127]   │
         │          │  │ cast → int8            │
         └────┬─────┘  └──────┬────────────────┘
              │               │
         ┌────▼───────────────▼──────────────────────┐
         │           CSR-style packed storage         │
         │                                            │
         │  int8_data   [total_int8]  int8            │
         │  fp16_data   [total_fp16]  fp16            │
         │  bitmask     [B, W]        int64           │
         │    W = ceil(BLOCK_SIZE / 64)               │
         │    bit i = 1 → position i is FP16         │
         │  scales      [B]           fp16            │
         │  int8_offsets[B+1]         int32  (CSR)   │
         │  fp16_offsets[B+1]         int32  (CSR)   │
         └───────────────────────────────────────────┘
```

---

### Quantization Pseudocode

```
function quantize_mixed(weight):
    flat   = weight.flatten().float()
    flat   = pad(flat, to_multiple_of=BLOCK_SIZE)
    blocks = flat.reshape(num_blocks, BLOCK_SIZE)

    for each block b:
        μ, σ = mean(blocks[b]), std(blocks[b])
        outlier_mask[b] = |blocks[b] − μ| > OUTLIER_SIGMA × σ

        # Scale from non-outlier values only
        safe   = blocks[b] with outliers zeroed out
        absmax = max(|safe|)
        scale[b] = absmax / 127.0           # stored as fp16

        # Quantize all positions (outlier slots get 0, ignored on decode)
        q[b] = round(blocks[b] / absmax × 127)
        q[b] = clamp(q[b], −127, 127)

    # Bitmask: ceil(BLOCK_SIZE/64) int64 words per block
    # Word w covers positions [w×64, (w+1)×64)
    # bit i = 1 means position i is an outlier (FP16)
    for word w in range(ceil(BLOCK_SIZE / 64)):
        bitmask[:, w] = OR of (1 << local_pos) for each outlier position

    # Pack into contiguous 1-D arrays (position-ascending order preserved)
    int8_data   = q[~outlier_mask]                  # non-outlier int8 values
    fp16_data   = blocks[outlier_mask].to(fp16)     # outlier fp16 values
    int8_offsets = cumsum(count of int8 values per block)
    fp16_offsets = cumsum(count of fp16 values per block)

    return MixedQuantWeight(int8_data, fp16_data, bitmask,
                            scales, int8_offsets, fp16_offsets,
                            original_shape)
```

---

### Dequantization Block Diagram

```
  MixedQuantWeight (stored buffers on device)
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │  Step 1 — Expand bitmask → bool mask [B, S]  │
  │                                              │
  │  For each word w in [0, W):                  │
  │    local_pos = arange(64)                    │
  │    part[b, lo:hi] = (bitmask[b,w] >> pos) & 1│
  │  fp16_mask = concat(parts)   [B, S]  bool    │
  │  int8_mask = ~fp16_mask                      │
  └──────────────────┬───────────────────────────┘
                     │
  ┌──────────────────▼───────────────────────────┐
  │  Step 2 — Flat scatter indices               │
  │                                              │
  │  all_pos  = block_start + position_in_block  │
  │  fp16_pos = all_pos[fp16_mask]               │
  │  int8_pos = all_pos[int8_mask]               │
  └──────────────────┬───────────────────────────┘
                     │
  ┌──────────────────▼───────────────────────────┐
  │  Step 3 — Scatter into flat fp16 buffer      │
  │                                              │
  │  flat[fp16_pos] = fp16_data       (verbatim) │
  │                                              │
  │  scale_per_val  = scales[block_idx]          │
  │  flat[int8_pos] = int8_data.fp16 × scale     │
  └──────────────────┬───────────────────────────┘
                     │
                     ▼
         Trim padding → reshape to original_shape
                     │
                     ▼
           fp16 weight tensor  [N, K]
```

---

### Dequantization Pseudocode

```
function dequantize_mixed(mqw):
    flat = zeros(num_blocks × BLOCK_SIZE, dtype=fp16)

    # Reconstruct bool mask from multi-word bitmask
    fp16_mask = []
    for word w in range(W):
        positions = arange(64)
        bits = (mqw.bitmask[:, w] >> positions) & 1
        fp16_mask.append(bits.bool())
    fp16_mask = concat(fp16_mask)       # [num_blocks, BLOCK_SIZE]
    int8_mask  = ~fp16_mask

    # Compute flat scatter positions
    all_pos   = block_starts + position_in_block   # [B, S]
    fp16_pos  = all_pos[fp16_mask]
    int8_pos  = all_pos[int8_mask]

    # Scatter FP16 outliers verbatim
    flat[fp16_pos] = mqw.fp16_data

    # Scatter dequantized INT8 values
    block_idx      = row index of each int8 position
    flat[int8_pos] = mqw.int8_data.float16 × mqw.scales[block_idx]

    return flat[:original_numel].reshape(original_shape)
```

---

### Forward Pass (MixedQuantLinear)

```
def forward(x):
    W = dequantize_mixed(self.stored_buffers)   # reconstruct fp16 weight
    return F.linear(x.fp16, W, bias)            # standard linear op
```

---

## Mode 2 — Pure INT8 (`QUANT_MODE = "pure_int8"`)

### What it does

No outlier tracking. Every value in a block is quantized to INT8 using a single per-block scale (`absmax`). Optionally, a **companding curve** (mu-law or A-law) is applied after normalization to redistribute quantization levels — giving more precision to small values where most weights cluster.

---

### Quantization Block Diagram

```
  Weight matrix  [N, K]  (fp32 working copy)
         │
         ▼
  ┌──────────────────────────────────────────┐
  │  Pad K to multiple of BLOCK_SIZE         │
  │  Reshape → [N, num_blocks, BLOCK_SIZE]   │
  └───────────────────┬──────────────────────┘
                      │
         ┌────────────▼──────────────┐
         │  Per-block absmax         │
         │  scales[n,b] = max(|w|)   │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────────────┐
         │  Normalize                        │
         │  w_norm = w / scales  ∈ [−1, 1]  │
         └────────────┬──────────────────────┘
                      │
         ┌────────────▼──────────────────────────────────┐
         │  Companding  (if COMPANDING ≠ "none")         │
         │                                               │
         │  mu-law:  y = sign(x) × log(1 + μ|x|)        │
         │                        ─────────────          │
         │                          log(1 + μ)           │
         │                                               │
         │  A-law:   y = A|x|/(1+ln A)    |x| < 1/A     │
         │           y = (1+ln A|x|)/(1+ln A) otherwise  │
         │                                               │
         │  Output is still ∈ [−1, 1]                   │
         └────────────┬──────────────────────────────────┘
                      │
         ┌────────────▼──────────────────────┐
         │  Quantize                         │
         │  q = round(w_comp × 127)          │
         │  q = clamp(q, −127, 127)  → int8  │
         └────────────┬──────────────────────┘
                      │
                      ▼
         Storage: qweight [N, K_pad]  int8
                  scales  [N, num_blocks]  fp16
```

---

### Quantization Pseudocode

```
function quantize_int8_block(weight):
    weight = pad_rows(weight, to_multiple_of=BLOCK_SIZE)
    w      = weight.float().reshape(N, num_blocks, BLOCK_SIZE)

    scales = max(|w|, dim=BLOCK_SIZE)             # absmax per block
    w_norm = w / scales                           # ∈ [−1, 1]
    w_comp = compand(w_norm)                      # identity if COMPANDING="none"

    qweight = round(w_comp × 127)
    qweight = clamp(qweight, −127, 127).to(int8)

    return qweight.reshape(N, K_pad), scales.to(fp16)


function compand(x):                              # mu-law example
    return sign(x) × log(1 + μ × |x|) / log(1 + μ)
```

---

### Dequantization Block Diagram

```
  qweight [N, K_pad]  int8
  scales  [N, num_blocks]  fp16
         │
         ▼
  ┌──────────────────────────────────────────┐
  │  Reshape → [N, num_blocks, BLOCK_SIZE]   │
  │  Cast to fp16, divide by 127             │
  │  w_comp ∈ [−1, 1]                        │
  └───────────────────┬──────────────────────┘
                      │
         ┌────────────▼──────────────────────────────────┐
         │  Inverse companding  (expand)                 │
         │                                               │
         │  mu-law:  x = sign(y) × ((1+μ)^|y| − 1) / μ │
         │                                               │
         │  A-law:   x = |y|(1+ln A)/A    |y| < 1/(1+lnA)│
         │           x = e^(|y|(1+ln A)−1)/A  otherwise  │
         │                                               │
         │  Output w_norm ∈ [−1, 1]                     │
         └────────────┬──────────────────────────────────┘
                      │
         ┌────────────▼──────────────────────┐
         │  Denormalize                      │
         │  w = w_norm × scales              │
         │  Trim padding → original shape    │
         └────────────┬──────────────────────┘
                      │
                      ▼
           fp16 weight tensor  [N, K]
```

---

### Dequantization Pseudocode

```
function dequantize_int8_block(qweight, scales, original_shape):
    w_comp = qweight.reshape(N, num_blocks, BLOCK_SIZE).fp16 / 127.0
    w_norm = expand(w_comp)                       # inverse of compand
    w      = w_norm × scales                      # denormalize
    return w.reshape(N, K_pad)[:, :original_K].reshape(original_shape)


function expand(y):                               # mu-law inverse
    return sign(y) × ((1 + μ)^|y| − 1) / μ
```

---

### Forward Pass (PureInt8Linear)

```
# Without USE_FUSED_MATMUL:
def forward(x):
    W = dequantize_int8_block(qweight, scales, shape)  # full fp16 weight matrix
    return F.linear(x.fp16, W, bias)

# With USE_FUSED_MATMUL (Plan C):
def forward(x):
    x_blocked = x.reshape(batch, num_blocks, BLOCK_SIZE)
    w_blocked = qweight.reshape(N, num_blocks, BLOCK_SIZE).fp16 × scales
    out = einsum("ibk,nbk->in", x_blocked.f32, w_blocked.f32).fp16
    #    ↑ accumulate in float32 to avoid fp16 overflow
    return out + bias
```

---

## Companding (pure\_int8 only)

Companding is a nonlinear preprocessing step borrowed from audio compression. It reshapes the [−1, 1] normalized weight distribution before quantization so that **more INT8 levels are allocated to small values** (where the bulk of transformer weights live) and fewer to large values.

```
Without companding             With mu-law companding
─────────────────────          ──────────────────────
INT8 levels spread uniformly   INT8 levels denser near 0
over [−1, 1]:                  where most weights cluster:

  −1    0    +1                   −1    0    +1
   |....|....|                     |.||||||..|
   ↑ wasted levels                 ↑ more precision here
     at the extremes
```

| Setting | Description |
|---|---|
| `COMPANDING = "none"` | Standard linear quantization |
| `COMPANDING = "mu_law"` | Logarithmic compression, `MU_LAW_MU=255` is the standard telecom value |
| `COMPANDING = "a_law"` | Piecewise linear+log, `A_LAW_A=87.6` is the ITU standard value |

**Important:** companding applies only to `pure_int8` mode. The `mixed` mode handles small values by design (they are rarely outliers, so they get a finer INT8 scale computed without the outliers skewing it).

---

## Inference Speed-ups

### Plan B — `torch.compile`

```python
USE_COMPILE = True
COMPILE_MODE = "default"   # or "reduce-overhead", "max-autotune"
```

Wraps `dequantize_mixed`, `dequantize_int8_block`, and `_int8_fused_matmul` with `torch.compile`. TorchInductor fuses element-wise ops (bitmask shifts, scale multiplies) into fewer CUDA kernels. The first call per layer triggers JIT compilation (warm-up overhead); all subsequent calls use the compiled graph.

`_int8_fused_matmul` is compiled with `dynamic=True` so that a single compiled graph handles all sequence lengths during autoregressive generation without recompiling on every decode step.

### Plan C — Fused block matmul (`USE_FUSED_MATMUL`, pure\_int8 only)

```python
USE_FUSED_MATMUL = True
```

Instead of fully dequantizing the weight to `[N, K]` fp16 and then calling `F.linear`, the forward pass reshapes both the input and the weight into a blocked layout and computes a single einsum:

```
x_blocked  [batch, num_blocks, BLOCK_SIZE]
w_blocked  [N,     num_blocks, BLOCK_SIZE]  ← int8 × scale (never flattened to [N,K])

out = einsum("ibk,nbk->in", x_blocked, w_blocked)
```

This avoids allocating the full `[N, K_pad]` fp16 weight tensor during every forward pass. Combined with `torch.compile`, TorchInductor can further fuse the INT8→fp32 cast and scale multiply into the GEMM.

---

## Memory Savings

For a weight matrix of shape `[N, K]` in fp16 (2 bytes/value):

| Mode | Bytes per weight value | Overhead |
|---|---|---|
| fp16 baseline | 2.00 B | — |
| `pure_int8` (BLOCK_SIZE=128) | ~1.02 B | 1 fp16 scale per 128 values |
| `mixed` (BLOCK_SIZE=128, ~5% outliers) | ~1.17 B | scale + bitmask + ~5% fp16 values |
| `mixed` (BLOCK_SIZE=128, ~0% outliers) | ~1.02 B | same as pure_int8 at the extreme |

Larger `BLOCK_SIZE` reduces overhead (fewer scales/bitmasks) but increases quantization error per block.

---

## Project Structure

```
quantize.py                 — main implementation
layers_to_quantize.txt      — list of layers to quantize
requirements.txt            — Python dependencies
```
