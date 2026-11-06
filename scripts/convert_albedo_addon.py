#!/usr/bin/env python3
"""Convert the albedo add-on for Qwen-Image 2.1 (a Marigold V2 style checkpoint: trainables.safetensors) into the
files sd_albedo / sd-cli --mode albedo read.

Input (--trainables): PEFT names, as the trainer saves them
    Diffuser.<diffusers module path>.lora_A.default.weight   [rank, in]
    Diffuser.<diffusers module path>.lora_B.default.weight   [out, rank]
    VAE.decoder.*, VAE.post_quant_conv.*                      the fine-tuned decoder

Output (--out-dir):
    albedo_lora.safetensors  diffusion_model.<path>.lora_A.weight / .lora_B.weight and a scalar .alpha per pair
                             (the LoRA loader's names; scale = alpha / rank)
    albedo_vae.safetensors   the decoder alone (decoder.*, post_quant_conv.*), named as in the stock VAE file
                             (--vae), whose shapes and types it must match

The prompt embedding (albedo_condition.safetensors, tensor "prompt_embeds" [L, 4096]) is made by the trainer's
scripts and is used as it is.

Needs torch and safetensors.
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LORA_PREFIX = "Diffuser."
VAE_PREFIX = "VAE."
DECODER_PARTS = ("decoder.", "post_quant_conv.")


def convert_lora(trainables, alpha):
    pairs = {}
    for key, tensor in trainables.items():
        if not key.startswith(LORA_PREFIX):
            continue
        path = key[len(LORA_PREFIX):]
        for side, suffix in (("A", ".lora_A.default.weight"), ("B", ".lora_B.default.weight")):
            if path.endswith(suffix):
                pairs.setdefault(path[: -len(suffix)], {})[side] = tensor
                break
        else:
            raise ValueError(f"unexpected LoRA tensor {key}")
    out = {}
    for module, pair in sorted(pairs.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError(f"{module}: lora_A and lora_B must come together, found {sorted(pair)}")
        down, up = pair["A"], pair["B"]
        if down.dim() != 2 or up.dim() != 2 or up.shape[1] != down.shape[0]:
            raise ValueError(f"{module}: shapes A {tuple(down.shape)} B {tuple(up.shape)} do not form a LoRA")
        rank = down.shape[0]
        name = "diffusion_model." + module
        out[name + ".lora_A.weight"] = down.contiguous()
        out[name + ".lora_B.weight"] = up.contiguous()
        out[name + ".alpha"] = torch.tensor(float(rank if alpha is None else alpha), dtype=torch.float32)
    return out, len(pairs)


def convert_decoder(trainables, vae_path):
    out = {}
    for key, tensor in trainables.items():
        if key.startswith(VAE_PREFIX):
            name = key[len(VAE_PREFIX):]
            if not name.startswith(DECODER_PARTS):
                raise ValueError(f"unexpected VAE tensor {key}: only the decoder is trained")
            out[name] = tensor.contiguous()
    with safe_open(vae_path, framework="pt") as stock:
        stock_names = {k for k in stock.keys() if k.startswith(DECODER_PARTS)}
        if stock_names != set(out):
            missing = sorted(stock_names - set(out))[:5]
            extra = sorted(set(out) - stock_names)[:5]
            raise ValueError(f"decoder tensors differ from {vae_path}: missing {missing}, extra {extra}")
        for name, tensor in out.items():
            ref = stock.get_slice(name)
            if list(ref.get_shape()) != list(tensor.shape):
                raise ValueError(f"{name}: shape {tuple(tensor.shape)} != stock {ref.get_shape()}")
            if ref.get_dtype() != "BF16" or tensor.dtype != torch.bfloat16:
                raise ValueError(f"{name}: expected BF16 like the stock VAE, got {tensor.dtype} / {ref.get_dtype()}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainables", required=True, type=Path)
    parser.add_argument("--vae", required=True, type=Path, help="the stock Qwen-Image 2.1 VAE (qwen_image_2.1_vae_bf16.safetensors)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha (default: the rank, scale 1)")
    parser.add_argument("--expect-pairs", type=int, default=229, help="LoRA pairs the checkpoint must hold (0: any)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    lora_out = args.out_dir / "albedo_lora.safetensors"
    vae_out = args.out_dir / "albedo_vae.safetensors"
    for path in (lora_out, vae_out):
        if path.exists() and not args.force:
            sys.exit(f"{path} exists (--force overwrites it)")

    trainables = load_file(str(args.trainables), device="cpu")
    lora, pairs = convert_lora(trainables, args.alpha)
    if args.expect_pairs and pairs != args.expect_pairs:
        sys.exit(f"{pairs} LoRA pairs in {args.trainables}, expected {args.expect_pairs}")
    decoder = convert_decoder(trainables, args.vae)
    leftover = [k for k in trainables if not k.startswith((LORA_PREFIX, VAE_PREFIX))]
    if leftover:
        sys.exit(f"tensors neither LoRA nor decoder: {leftover[:5]}")

    # one metadata entry: safetensors writes several in hash order, which changed the header between runs
    meta = {"format": f"sd.cpp albedo add-on from {args.trainables.name}"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(lora, str(lora_out), metadata=meta)
    save_file(decoder, str(vae_out), metadata=meta)
    print(f"{lora_out}: {pairs} pairs, {len(lora)} tensors, {lora_out.stat().st_size:,} bytes")
    print(f"{vae_out}: {len(decoder)} tensors, {vae_out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
