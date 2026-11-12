#!/usr/bin/env python3
"""
Quantize TRELLIS.2 DiT flow checkpoints to INT8 with ConvRot (Hadamard rotation)
compatible with ComfyUI's MixedPrecisionOps / MixedPrecisionSparseOps and comfy-kitchen.

Preserves in BF16:
  - Input and output stems (input_layer, out_layer)
  - Timestep embedding & global AdaLN modulation (t_embedder, adaLN_modulation)
  - Per-block modulation vectors, LayerNorms, and QK RMSNorms
  - Optional boundary blocks (e.g. blocks.0 and blocks.29 if --keep-boundary-blocks is set)

Quantizes to TensorWiseINT8Layout (per_channel=True, convrot=True, convrot_groupsize=256):
  - blocks.{i}.self_attn.{to_qkv, to_out}
  - blocks.{i}.cross_attn.{to_q, to_kv, to_out}
  - blocks.{i}.mlp.mlp.{0, 2}
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kuroko/ComfyUI")

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy.quant_ops import QuantizedTensor


SKIP_PREFIXES_BASE = (
    "t_embedder.",
    "adaLN_modulation.",
    "input_layer.",
    "out_layer.",
)

SKIP_SUBSTRINGS = (
    "modulation",
    "norm",
    "rms_norm",
    "pos_embedder",
    "rope",
)


def should_quantize(
    key: str,
    tensor: torch.Tensor,
    group_size: int = 256,
    keep_boundary_blocks: bool = False,
    num_blocks: int = 30,
) -> tuple[bool, str]:
    if not key.endswith(".weight"):
        return False, "not_weight"
    if tensor.ndim != 2:
        return False, f"ndim={tensor.ndim}"
    if not key.startswith("blocks."):
        return False, "non_block"
    for p in SKIP_PREFIXES_BASE:
        if key.startswith(p):
            return False, f"skip_prefix({p})"
    for s in SKIP_SUBSTRINGS:
        if s in key:
            return False, f"skip_substr({s})"
    if keep_boundary_blocks:
        parts = key.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            b_idx = int(parts[1])
            if b_idx == 0 or b_idx == num_blocks - 1:
                return False, f"boundary_block({b_idx})"
    out_features, in_features = tensor.shape
    if in_features % group_size != 0:
        return False, f"in_features={in_features}_not_div_by_{group_size}"
    return True, "ok"


def quantize_checkpoint(
    src_path: str,
    dst_path: str,
    group_size: int = 256,
    keep_boundary_blocks: bool = False,
    device: str = "cuda",
) -> dict:
    t0 = time.perf_counter()
    out_sd = {}
    quantized_layers = 0
    skipped_layers = 0
    orig_bytes = 0
    new_bytes = 0

    src_json = src_path.replace(".safetensors", ".json")
    num_blocks = 30
    if os.path.exists(src_json):
        with open(src_json, "r") as f:
            cfg = json.load(f)
            num_blocks = cfg.get("args", {}).get("num_blocks", 30)

    print(f"[Quantize] Loading {src_path} (num_blocks={num_blocks}, keep_boundary={keep_boundary_blocks})...")
    with safe_open(src_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            tensor = f.get_tensor(k)
            orig_bytes += tensor.numel() * tensor.element_size()

            ok, reason = should_quantize(
                k, tensor, group_size=group_size,
                keep_boundary_blocks=keep_boundary_blocks, num_blocks=num_blocks,
            )
            if not ok:
                if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                    tensor = tensor.to(torch.bfloat16)
                out_sd[k] = tensor
                new_bytes += tensor.numel() * tensor.element_size()
                if k.endswith(".weight"):
                    skipped_layers += 1
                continue

            prefix = k[:-len("weight")]
            w_gpu = tensor.to(device=device, dtype=torch.bfloat16)
            qt = QuantizedTensor.from_float(
                w_gpu,
                "TensorWiseINT8Layout",
                per_channel=True,
                convrot=True,
                convrot_groupsize=group_size,
            )
            q_sd = qt.state_dict(f"{prefix}weight")
            for qk, qv in q_sd.items():
                qv_cpu = qv.detach().cpu().contiguous()
                out_sd[qk] = qv_cpu
                new_bytes += qv_cpu.numel() * qv_cpu.element_size()

            quant_meta = {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": group_size,
            }
            meta_tensor = torch.tensor(
                list(json.dumps(quant_meta).encode("utf-8")),
                dtype=torch.uint8,
            )
            out_sd[f"{prefix}comfy_quant"] = meta_tensor
            new_bytes += meta_tensor.numel() * meta_tensor.element_size()
            quantized_layers += 1

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    save_file(out_sd, dst_path)

    dst_json = dst_path.replace(".safetensors", ".json")
    if os.path.exists(src_json) and os.path.abspath(src_json) != os.path.abspath(dst_json):
        shutil.copy2(src_json, dst_json)

    elapsed = time.perf_counter() - t0
    stats = {
        "src": src_path,
        "dst": dst_path,
        "quantized_layers": quantized_layers,
        "skipped_weight_layers": skipped_layers,
        "orig_mb": orig_bytes / (1024 ** 2),
        "new_mb": new_bytes / (1024 ** 2),
        "compression_ratio": orig_bytes / max(new_bytes, 1),
        "elapsed_s": elapsed,
    }
    print(
        f"[Quantize] Saved {dst_path}: "
        f"{quantized_layers} layers quantized, {skipped_layers} weight layers kept in BF16, "
        f"{stats['orig_mb']:.1f} MB -> {stats['new_mb']:.1f} MB ({stats['compression_ratio']:.2f}x), "
        f"{elapsed:.1f}s"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Quantize TRELLIS.2 flow checkpoints to INT8 ConvRot")
    parser.add_argument("--src-dir", type=str, default="/mnt/data/backup/comfyui/ckpts")
    parser.add_argument("--dst-dir", type=str, default="/mnt/data/ComfyUI/models/trellis2/ckpts")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--keep-boundary-blocks", action="store_true", help="Keep block 0 and last block in BF16")
    parser.add_argument("--only", type=str, default=None, help="Substring filter for checkpoint name")
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    flow_models = [
        "ss_flow_img_dit_1_3B_64_bf16",
        "slat_flow_img2shape_dit_1_3B_512_bf16",
        "slat_flow_img2shape_dit_1_3B_1024_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    ]

    for name in flow_models:
        if args.only and args.only not in name:
            continue
        src_path = os.path.join(args.src_dir, f"{name}.safetensors")
        if not os.path.exists(src_path):
            print(f"[Quantize] Skipping missing {src_path}")
            continue
        stem = name[:-5] if name.endswith("_bf16") else name
        dst_path = os.path.join(args.dst_dir, f"{stem}_int8_convrot.safetensors")
        quantize_checkpoint(
            src_path=src_path,
            dst_path=dst_path,
            group_size=args.group_size,
            keep_boundary_blocks=args.keep_boundary_blocks,
        )


if __name__ == "__main__":
    main()
