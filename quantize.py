"""
LLaMA/Qwen quantisation runner.

Set QUANT_MODE in config.py to select the quantisation strategy, then run:
    python quantize.py
"""

import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_ID, MAX_NEW, LAYERS_FILE, QUANT_MODE
from mixed import MixedQuantLinear
from pure_int8 import PureInt8Linear
from col_int8 import ColOutlierLinear


_MODE_CLS = {
    "mixed":     MixedQuantLinear,
    "pure_int8": PureInt8Linear,
    "col_int8":  ColOutlierLinear,
}


def load_layer_names(path: str) -> list[str]:
    names = []
    with open(path) as fh:
        for line in fh:
            name = line.split("#")[0].strip()
            if name:
                names.append(name)
    return names


def quantize_selected_layers(model: nn.Module, layers_file: str) -> nn.Module:
    cls   = _MODE_CLS[QUANT_MODE]
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
        setattr(parent, attr, cls(child))
        print(f"  [done] {full_name}  ({QUANT_MODE})")
    return model


def model_size_mb(model: nn.Module) -> float:
    params  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (params + buffers) / 1024 ** 2


def count_layers(model: nn.Module) -> tuple[int, int]:
    quant_types = (MixedQuantLinear, PureInt8Linear, ColOutlierLinear)
    quant = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, *quant_types)):
            total += 1
            if isinstance(m, quant_types):
                quant += 1
    return quant, total


def _first_non_meta_param_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        if getattr(p, "device", None) is not None and str(p.device) != "meta":
            return p.device
    return torch.device("cpu")


def compute_loss(model: nn.Module, tokenizer, text: str) -> float:
    batch  = tokenizer(text, return_tensors="pt").to(_first_non_meta_param_device(model))
    with torch.no_grad():
        outputs = model(**batch, labels=batch["input_ids"])
    return outputs.loss.item()


def main() -> None:
    texts  = ["Digital signal processing improves efficiency.", "Quantization reduces model size."]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}  |  mode : {QUANT_MODE}\n")

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_f16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).eval()
    size_fp16 = model_size_mb(model_f16)
    print(f"  Float16 size      : {size_fp16:,.1f} MB")

    loss_fp16 = compute_loss(model_f16, tokenizer, texts[0])
    print(f"  FP16 loss         : {loss_fp16:.4f}")

    model_f16.cpu()
    del model_f16
    torch.cuda.empty_cache()

    print(f"\nLoading {MODEL_ID} for quantisation ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).eval()

    print(f"\nQuantising layers from '{LAYERS_FILE}' ...")
    t0 = time.time()
    with torch.no_grad():
        quantize_selected_layers(model, LAYERS_FILE)
    dt = time.time() - t0

    size_q, total = count_layers(model)
    print(f"  Done in           : {dt:.1f} s")
    print(f"  Quantised size    : {model_size_mb(model):,.1f} MB  (was {size_fp16:,.1f} MB)")
    print(f"  Layers quantised  : {size_q} / {total}")

    model = model.to(device)

    loss_q = compute_loss(model, tokenizer, texts[0])
    print(f"  Quant loss        : {loss_q:.4f}")

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


if __name__ == "__main__":
    main()
