#!/usr/bin/env python3
# Copyright 2026 bitserv-ai
"""
Quantize Qwen3-VL Embedding and Reranker models to W8A16 (weight-only INT8).

Uses llmcompressor's QuantizationModifier (RTN) with scheme="W8A16".
RTN (round-to-nearest) requires NO calibration data — weight-only quantization
computes per-channel scales directly from weight statistics.

CRITICAL NOTES (from code review, source-code verified):
  - QuantizationModifier does NOT compute Hessians (unlike GPTQModifier).
    GPTQModifier ALWAYS accumulates Hessians even for W8A16 — use
    QuantizationModifier for calibration-free weight-only RTN.
  - torch_dtype=torch.bfloat16 is forced for BOTH models. The Reranker's
    config.json declares float32, but actual safetensors weights are BF16
    (the checkpoint is 17 GiB, not 32 GiB). Forcing BF16 halves RAM usage.
  - device_map is left default (auto) because llmcompressor's DataFreePipeline
    uses dispatch_model() which requires a visible device. On CPU-only systems,
    the model stays on CPU automatically. On systems with a GPU, the model may
    be placed on GPU during quantization — stop vLLM/Lemonade before running.
  - The Reranker checkpoint does NOT contain a "score" layer. The score head
    is injected at runtime by vLLM via --hf-overrides
    (Qwen3VLForSequenceClassification). Both models use the same ignore list.
  - oneshot() automatically saves model + processor with save_compressed=True.
    No separate model.save_pretrained() call is needed.
  - AutoModelForImageTextToText resolves to Qwen3VLForConditionalGeneration
    for model_type "qwen3_vl", preserving multimodal weight prefixes.

Output format: compressed-tensors (safetensors), directly loadable by vLLM
with --quantization compressed-tensors.

Prerequisites:
  python3.12 -m venv /opt/src/vllm/.venv-quantize
  source /opt/src/vllm/.venv-quantize/bin/activate
  pip install llmcompressor==0.10.0.1 torch transformers compressed-tensors

  Note: CPU-only PyTorch is sufficient. No GPU required.
  ~30 GiB RAM per model (BF16 weights loaded on CPU).

Usage:
  source /opt/src/vllm/.venv-quantize/bin/activate
  python quantize_w8a16.py --model embedding
  python quantize_w8a16.py --model reranker
  python quantize_w8a16.py --model both
"""

import argparse
import json
import os
import sys
import time

import torch

# CPU-only torch cannot dispatch models across devices.
# Monkey-patch get_device_memory to report CPU memory so dispatch_model
# places everything on CPU instead of raising MemoryError.
from compressed_tensors.offload.dispatch import get_device_memory as _orig_get_device_memory

def _get_device_memory_with_cpu():
    mem = _orig_get_device_memory()
    if len(mem) == 0:
        import psutil
        available = int(psutil.virtual_memory().available)
        mem[torch.device("cpu")] = available
    return mem

import compressed_tensors.offload.dispatch
compressed_tensors.offload.dispatch.get_device_memory = _get_device_memory_with_cpu

from transformers import AutoModelForImageTextToText, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Base path for models; set VLLM_MODEL_HOME before running if needed.
default_model_path = os.environ.get("VLLM_MODEL_HOME", "/path/to/your/models")

MODELS = {
    "embedding": {
        "input": os.path.join(default_model_path, "Qwen3-VL-Embedding-8B"),
        "output": os.path.join(default_model_path, "Qwen3-VL-Embedding-8B-W8A16"),
        "ignore": ["lm_head", "re:.*visual.*"],
    },
    "reranker": {
        "input": os.path.join(default_model_path, "Qwen3-VL-Reranker-8B"),
        "output": os.path.join(default_model_path, "Qwen3-VL-Reranker-8B-W8A16"),
        "ignore": ["lm_head", "re:.*visual.*"],
    },
}

DTYPE = torch.bfloat16


def verify_source_dtype(model_dir):
    from safetensors import safe_open
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"  WARNING: No safetensors index found at {index_path}")
        print(f"  Skipping dtype verification. Proceeding with forced {DTYPE}.")
        return
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    lm_head_dtype = weight_map.get("lm_head.weight")
    if lm_head_dtype:
        shard_file = os.path.join(model_dir, lm_head_dtype)
        if os.path.exists(shard_file):
            with safe_open(shard_file, framework="pt") as sf:
                dtype = sf.get_tensor("lm_head.weight").dtype
                print(f"  Source lm_head dtype: {dtype}")
                if dtype not in (torch.bfloat16, torch.float16):
                    print(f"  WARNING: Expected BF16/FP16 on disk, got {dtype}.")
                    print(f"  The config may declare float32 but weights are {dtype}.")
    print(f"  Forcing torch_dtype={DTYPE} for quantization.")


def verify_output(output_dir, model_name):
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"  ERROR: {config_path} not found after quantization!")
        return False
    with open(config_path) as f:
        config = json.load(f)
    qconfig = config.get("quantization_config", {})
    if not qconfig:
        print(f"  ERROR: No quantization_config in output config.json!")
        return False
    groups = qconfig.get("config_groups", {})
    if "group_0" not in groups:
        print(f"  ERROR: No group_0 in quantization_config!")
        return False
    g0 = groups["group_0"]
    weights = g0.get("weights", {})
    if weights.get("num_bits") != 8:
        print(f"  ERROR: Expected num_bits=8, got {weights.get('num_bits')}")
        return False
    if weights.get("type") != "int":
        print(f"  ERROR: Expected type='int', got {weights.get('type')}")
        return False
    if g0.get("input_activations") is not None:
        print(f"  WARNING: input_activations is not None — this is W8A8, not W8A16!")
        print(f"  input_activations = {g0['input_activations']}")
        return False
    ignored = qconfig.get("ignore", [])
    has_visual = any("visual" in str(i) for i in ignored)
    has_lm_head = "lm_head" in ignored
    if not has_visual:
        print(f"  WARNING: No 'visual' pattern in ignore list — ViT may be quantized!")
    if not has_lm_head:
        print(f"  WARNING: 'lm_head' not in ignore list — prediction head may be quantized!")
    print(f"  quant_method: {qconfig.get('quant_method')}")
    print(f"  scheme: W{weights['num_bits']}A16 (weight-only INT8)")
    print(f"  ignore: {ignored}")
    print(f"  weights: num_bits={weights['num_bits']}, type={weights['type']}, "
          f"strategy={weights.get('strategy')}, symmetric={weights.get('symmetric')}")
    return True


def quantize_model(name, cfg):
    input_dir = cfg["input"]
    output_dir = cfg["output"]
    ignore = cfg["ignore"]

    if not os.path.isdir(input_dir):
        print(f"ERROR: Source model not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Model:    {name}")
    print(f"  Source:   {input_dir}")
    print(f"  Output:   {output_dir}")
    print(f"  Scheme:   W8A16 (weight-only INT8, RTN)")
    print(f"  Modifier: QuantizationModifier (not GPTQModifier)")
    print(f"  DType:    {DTYPE} (forced, overrides config)")
    print(f"  Device:   auto (stop vLLM/Lemonade before running)")
    print(f"  Ignore:   {ignore}")
    print(f"{'='*60}")

    if os.path.isdir(output_dir) and os.listdir(output_dir):
        print(f"\nWARNING: Output dir already exists and is non-empty: {output_dir}")
        print(f"  Files may be overwritten. Continuing in 5s... (Ctrl+C to abort)")
        time.sleep(5)
    else:
        os.makedirs(output_dir, exist_ok=True)

    print("\n[0/4] Verifying source model dtype...")
    verify_source_dtype(input_dir)

    print(f"\n[1/4] Loading model ({DTYPE})...")
    model = AutoModelForImageTextToText.from_pretrained(
        input_dir,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Move model to a single device for dispatch_model compatibility.
    # CPU-only torch cannot dispatch across devices. If CUDA is available,
    # use GPU; otherwise place on CPU and provide device_memory manually.
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    print("[2/4] Loading processor (tokenizer + image preprocessor)...")
    processor = AutoProcessor.from_pretrained(input_dir, trust_remote_code=True)

    print("[3/4] Applying W8A16 quantization (QuantizationModifier, RTN, no calibration)...")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W8A16",
        ignore=ignore,
    )

    oneshot(
        model=model,
        recipe=recipe,
        output_dir=output_dir,
    )

    print("[4/4] Verifying output...")
    if not verify_output(output_dir, name):
        print(f"\nERROR: Output verification failed for {name}!")
        print(f"  Check {output_dir}/config.json manually.")
        sys.exit(1)

    print(f"\nDone! Quantized model saved to: {output_dir}")
    print(f"Load with: vllm serve {output_dir} --quantization compressed-tensors")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3-VL models to W8A16 INT8 (weight-only, RTN)"
    )
    parser.add_argument(
        "--model",
        choices=["embedding", "reranker", "both"],
        required=True,
        help="Which model(s) to quantize",
    )
    args = parser.parse_args()

    if args.model == "both":
        for name, cfg in MODELS.items():
            quantize_model(name, cfg)
            print()
    else:
        quantize_model(args.model, MODELS[args.model])


if __name__ == "__main__":
    main()
