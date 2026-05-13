# ── Global configuration ──────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_SIZE    = 1024     # values per quantisation block
OUTLIER_SIGMA = 3.0      # mixed mode: positions beyond this many σ → float16
MAX_NEW       = 50
LAYERS_FILE   = "layers_to_quantize.txt"

# Quantisation mode — choose one:
#   "mixed"    : outliers stored as float16 + bitmask per block (MixedQuantLinear)
#   "pure_int8": 1 float16 scale + BLOCK_SIZE int8 values per block (PureInt8Linear)
#   "col_int8" : outlier columns kept as float16, remaining columns block-int8 (ColOutlierLinear)
QUANT_MODE = "pure_int8"

# ── Inference speed-ups ───────────────────────────────────────────────────────
USE_COMPILE      = True       # Plan B: wrap hot paths with torch.compile
COMPILE_MODE     = "default"  # "default" | "reduce-overhead" | "max-autotune"
USE_FUSED_MATMUL = True       # Plan C: fused block einsum instead of dequant + F.linear

# ── Companding (pure_int8 / col_int8 only) ───────────────────────────────────
COMPANDING = "none"   # "none" | "mu_law" | "a_law"
MU_LAW_MU  = 255.0      # μ parameter for mu-law
A_LAW_A    = 87.6       # A parameter for A-law

# ── Column outlier detection (col_int8 only) ──────────────────────────────────
COL_OUTLIER_SIGMA = 3.0  # columns with absmax > mean + σ × std are kept as fp16
