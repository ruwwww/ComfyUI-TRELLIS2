#!/usr/bin/env python3
"""
Verification & Benchmark Suite for TRELLIS.2 ConvRot INT8 + comfy-kitchen Ops.

Tests:
1. SparseStructureFlowModel (Dense 3D DiT, 1.3B)
2. SLatFlowModel Shape (Sparse 3D DiT, 1.3B)
3. SLatFlowModel Texture (Sparse 3D DiT with concat_cond, 1.3B)

Verifies:
- QuantizedTensor residency (TensorWiseINT8Layout, convrot=True, group_size=256, int8 storage)
- Active comfy-kitchen kernel invocations (int8_linear, adaln, apply_rope)
- Numerical parity vs BF16 reference (Cosine Similarity, Relative L2, SNR dB)
- GPU VRAM footprint (MB) and Forward Pass Latency (ms/step)
"""

import gc
import os
import sys
import time

sys.path.insert(0, "/home/kuroko/ComfyUI")
sys.path.insert(0, "/home/kuroko/comfy-lab/skill/ComfyUI-TRELLIS2")

import torch
import comfy.model_management
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor
from comfy.sparse import SparseTensor

import nodes.stages as stages


def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict:
    r = ref.detach().float().flatten()
    t = test.detach().float().flatten()
    diff = r - t
    cos = torch.nn.functional.cosine_similarity(r.unsqueeze(0), t.unsqueeze(0), dim=-1).item()
    rel_l2 = (diff.norm() / (r.norm() + 1e-12)).item()
    snr_db = (10.0 * torch.log10((r.pow(2).mean() + 1e-12) / (diff.pow(2).mean() + 1e-12))).item()
    max_abs = diff.abs().max().item()
    return {
        "cosine_sim": cos,
        "rel_l2": rel_l2,
        "snr_db": snr_db,
        "max_abs": max_abs,
    }


def audit_quantized_layers(model: torch.nn.Module) -> dict:
    q_count = 0
    bf16_count = 0
    convrot_count = 0
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None:
            continue
        if isinstance(w, QuantizedTensor):
            q_count += 1
            params = getattr(w, "_params", None)
            if getattr(params, "convrot", False) and getattr(params, "convrot_groupsize", 0) == 256 and w._qdata.dtype == torch.int8:
                convrot_count += 1
        elif isinstance(w, torch.Tensor):
            bf16_count += 1
    return {
        "quantized_layers": q_count,
        "convrot_int8_layers": convrot_count,
        "bf16_layers": bf16_count,
    }


class CKOpCounter:
    def __init__(self):
        self.counts = {
            "int8_linear": 0,
            "int8_linear_fused_act": 0,
            "adaln": 0,
            "apply_rope": 0,
        }
        self._orig = {}
        self._orig_dispatch = None

    def __enter__(self):
        ck = comfy.quant_ops.ck
        for name in ("int8_linear", "adaln", "apply_rope"):
            orig_fn = getattr(ck, name)
            self._orig[name] = orig_fn

            def make_wrapper(n, fn):
                def wrapped(*args, **kwargs):
                    if n == "int8_linear":
                        self.counts["int8_linear_fused_act"] += 1
                    self.counts[n] += 1
                    return fn(*args, **kwargs)
                return wrapped

            setattr(ck, name, make_wrapper(name, orig_fn))

        import comfy_kitchen.tensor.base as ck_base
        layout_cls = comfy.quant_ops.TensorWiseINT8Layout
        self._orig_dispatch = []
        for aten_op in (
            torch.ops.aten.linear.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.addmm.default,
        ):
            op_table = ck_base._LAYOUT_DISPATCH_TABLE.get(aten_op)
            if op_table and layout_cls in op_table:
                orig_dispatch = op_table[layout_cls]
                self._orig_dispatch.append((op_table, layout_cls, orig_dispatch))

                def make_dispatch_wrapper(fn):
                    def wrapped_dispatch(qt, args, kwargs):
                        self.counts["int8_linear"] += 1
                        return fn(qt, args, kwargs)
                    return wrapped_dispatch

                op_table[layout_cls] = make_dispatch_wrapper(orig_dispatch)
        return self

    def __exit__(self, *exc):
        ck = comfy.quant_ops.ck
        for name, orig_fn in self._orig.items():
            setattr(ck, name, orig_fn)
        if self._orig_dispatch:
            for op_table, layout_cls, fn in self._orig_dispatch:
                op_table[layout_cls] = fn


def reset_stages_cache():
    for k in list(stages._model_patchers.keys()):
        stages._unload_model(k)
    stages._pipeline_config = None
    stages._model_paths.clear()
    stages._model_patchers.clear()
    stages._post_loaded.clear()
    gc.collect()
    torch.cuda.empty_cache()


def benchmark_fn(fn, warmup=2, iters=5):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    return out, ms


def verify_model(model_key: str, make_inputs_fn, extract_tensor_fn):
    print(f"\n{'=' * 72}")
    print(f"Verifying {model_key}")
    print(f"{'=' * 72}")

    # 1. BF16 Baseline
    os.environ["TRELLIS2_FORCE_BF16"] = "1"
    reset_stages_cache()
    stages._init_config()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    model_bf16 = stages._load_model(model_key)
    bf16_weight_mb = (torch.cuda.memory_allocated() - base_mem) / (1024 ** 2)
    audit_bf16 = audit_quantized_layers(model_bf16)

    inputs = make_inputs_fn()
    with torch.no_grad():
        out_bf16, ms_bf16 = benchmark_fn(lambda: model_bf16(*inputs), warmup=2, iters=5)
    ref_tensor = extract_tensor_fn(out_bf16).detach().clone()
    bf16_peak_mb = (torch.cuda.max_memory_allocated() - base_mem) / (1024 ** 2)
    del model_bf16, out_bf16
    reset_stages_cache()

    # 2. ConvRot INT8 + comfy-kitchen
    os.environ["TRELLIS2_FORCE_BF16"] = "0"
    stages._init_config()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    model_int8 = stages._load_model(model_key)
    int8_weight_mb = (torch.cuda.memory_allocated() - base_mem) / (1024 ** 2)
    audit_int8 = audit_quantized_layers(model_int8)

    inputs_int8 = make_inputs_fn()
    with torch.no_grad():
        with CKOpCounter() as counter:
            _ = model_int8(*inputs_int8)
        op_counts = dict(counter.counts)
        out_int8, ms_int8 = benchmark_fn(lambda: model_int8(*inputs_int8), warmup=2, iters=5)
    test_tensor = extract_tensor_fn(out_int8).detach().clone()
    int8_peak_mb = (torch.cuda.max_memory_allocated() - base_mem) / (1024 ** 2)
    del model_int8, out_int8
    reset_stages_cache()

    metrics = compute_metrics(ref_tensor, test_tensor)
    print(f"  [Audit BF16]   {audit_bf16}")
    print(f"  [Audit INT8]   {audit_int8}")
    print(f"  [CK Ops/Step]  {op_counts}")
    print(f"  [Weights VRAM] BF16: {bf16_weight_mb:.1f} MB -> INT8 ConvRot: {int8_weight_mb:.1f} MB ({bf16_weight_mb / max(int8_weight_mb, 1):.2f}x smaller)")
    print(f"  [Peak VRAM]    BF16: {bf16_peak_mb:.1f} MB -> INT8 ConvRot: {int8_peak_mb:.1f} MB")
    print(f"  [Latency]      BF16: {ms_bf16:.2f} ms -> INT8 ConvRot: {ms_int8:.2f} ms ({ms_bf16 / ms_int8:.2f}x speedup)")
    print(f"  [Parity]       Cosine Sim: {metrics['cosine_sim']:.6f} | Rel L2: {metrics['rel_l2']:.4f} | SNR: {metrics['snr_db']:.2f} dB")

    assert audit_int8["convrot_int8_layers"] == 210, f"Expected 210 ConvRot INT8 layers, got {audit_int8}"
    assert op_counts["int8_linear"] == 210, f"Expected 210 int8_linear calls per step, got {op_counts}"
    assert op_counts["int8_linear_fused_act"] == 30, f"Expected 30 fused linear_input_act calls per step, got {op_counts}"
    assert op_counts["adaln"] == 60, f"Expected 60 adaln calls per step, got {op_counts}"
    assert op_counts["apply_rope"] == 30, f"Expected 30 apply_rope calls per step, got {op_counts}"
    assert metrics["cosine_sim"] > 0.99, f"Cosine similarity too low: {metrics['cosine_sim']}"
    return {
        "model_key": model_key,
        "bf16_weight_mb": bf16_weight_mb,
        "int8_weight_mb": int8_weight_mb,
        "bf16_peak_mb": bf16_peak_mb,
        "int8_peak_mb": int8_peak_mb,
        "ms_bf16": ms_bf16,
        "ms_int8": ms_int8,
        "op_counts": op_counts,
        "metrics": metrics,
    }


def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Prepare deterministic inputs
    ss_x = torch.randn(1, 8, 16, 16, 16, device=device, dtype=torch.bfloat16)
    ss_t = torch.tensor([500.0], device=device, dtype=torch.bfloat16)
    ss_cond = torch.randn(1, 1029, 1024, device=device, dtype=torch.bfloat16)

    # Sparse coordinates (8192 active voxels in a 32^3 grid)
    N_vox = 8192
    grid_coords = torch.stack(
        torch.meshgrid(
            torch.arange(16, device=device, dtype=torch.int32),
            torch.arange(16, device=device, dtype=torch.int32),
            torch.arange(32, device=device, dtype=torch.int32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)[:N_vox]
    batch_col = torch.zeros((N_vox, 1), device=device, dtype=torch.int32)
    coords = torch.cat([batch_col, grid_coords], dim=-1)
    shape_feats = torch.randn(N_vox, 32, device=device, dtype=torch.bfloat16)
    tex_feats = torch.randn(N_vox, 32, device=device, dtype=torch.bfloat16)

    results = []

    # 1. SparseStructureFlowModel
    results.append(
        verify_model(
            "sparse_structure_flow_model",
            lambda: (ss_x.clone(), ss_t.clone(), ss_cond.clone()),
            lambda out: out,
        )
    )

    # 2. SLatFlowModel (Shape 512)
    results.append(
        verify_model(
            "shape_slat_flow_model_512",
            lambda: (
                SparseTensor(feats=shape_feats.clone(), coords=coords.clone()),
                ss_t.clone(),
                ss_cond.clone(),
            ),
            lambda out: out.feats,
        )
    )

    # 3. SLatFlowModel (Texture 512 with concat_cond)
    results.append(
        verify_model(
            "tex_slat_flow_model_512",
            lambda: (
                SparseTensor(feats=tex_feats.clone(), coords=coords.clone()),
                ss_t.clone(),
                ss_cond.clone(),
                SparseTensor(feats=shape_feats.clone(), coords=coords.clone()),
            ),
            lambda out: out.feats,
        )
    )

    print("\nAll TRELLIS.2 ConvRot INT8 + comfy-kitchen verifications PASSED!")


if __name__ == "__main__":
    main()
