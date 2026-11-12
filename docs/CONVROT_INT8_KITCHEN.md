# Technical Documentation: ConvRot INT8 Quantization & `comfy-kitchen` Kernel Integration for TRELLIS.2

This document describes the quantization methodology, `comfy-kitchen` kernel integration, sparse tensor optimizations, and verification results for running **TRELLIS.2** (Structure, Shape, and Texture 1.3B 3D DiT models) in ComfyUI.

---

## 1. Attribution & Prior Art

This implementation combines work across three foundational projects:

1. **Microsoft Research — [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) & [TRELLIS](https://github.com/microsoft/TRELLIS)**
   - Designed the 3D Sparse Structure Flow DiT (`SparseStructureFlowModel`), Structured Latent Flow DiT (`SLatFlowModel` for shape and PBR texture synthesis), and O-Voxel FlexiDualGrid representations.
2. **Andrea Pozzetti ([@PozzettiAndrea](https://github.com/PozzettiAndrea)) — [`ComfyUI-TRELLIS2`](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2)** (with contributions from **[@Fristender](https://github.com/Fristender)**)
   - Created the ComfyUI custom node architecture, stage-by-stage on-demand model loader (`nodes/stages.py`), native `ModelPatcher`/`KSampler` nodes (`nodes/nodes_native_sampling.py`), and consolidated TRELLIS.2 model definitions (`nodes/trellis2/`).
3. **ComfyOrg — [ComfyUI](https://github.com/comfyanonymous/ComfyUI) & `comfy-kitchen`**
   - **[@comfyanonymous](https://github.com/comfyanonymous)**: `comfy-kitchen` backend (`int8_linear`, `adaln`, `apply_rope`, `TensorWiseINT8Layout`, and Walsh-Hadamard `ConvRot` rotation kernels), `MixedPrecisionOps`, and TRELLIS.2 memory optimizations.
   - **Jukka Seppänen ([@kijai](https://github.com/kijai))**: ComfyUI's native `comfy/ldm/trellis2`, `comfy/sparse.py`, `comfy/ops_sparse.py`, and `int8_convrot` format integrations.
   - **[@rattus128](https://github.com/rattus128)**: `comfy.ops.linear_input_act` prologue/epilogue fusion (`gelu_tanh` + `ConvRot` + INT8 GEMM + gated residual epilogue).
   - **Jialong (Bruce) Li ([@chelsealong](https://github.com/chelsealong))**, **[@liminfei-amd](https://github.com/liminfei-amd)**, **Tijana Vukovic ([@tvukovic-amd](https://github.com/tvukovic-amd))**, and **[@DELUXA](https://github.com/DELUXA)**: `comfy.quant_ops` and `comfy-kitchen` dispatch enhancements.

---

## 2. Selective ConvRot INT8 Quantization Policy

TRELLIS.2 uses five 1.3B-parameter 30-block Diffusion Transformer (DiT) models (`model_channels = 1536`, `num_heads = 12`, `head_dim = 128`, `ffn_hidden = 6144`, `cond_channels = 1024`):

1. `ss_flow_img_dit_1_3B_64` (`SparseStructureFlowModel` — Dense 3D DiT)
2. `slat_flow_img2shape_dit_1_3B_512` (`SLatFlowModel` — Sparse 3D DiT, 512px)
3. `slat_flow_img2shape_dit_1_3B_1024` (`SLatFlowModel` — Sparse 3D DiT, 1024px cascade)
4. `slat_flow_imgshape2tex_dit_1_3B_512` (`SLatFlowModel` — Sparse 3D PBR Texture DiT, 512px)
5. `slat_flow_imgshape2tex_dit_1_3B_1024` (`SLatFlowModel` — Sparse 3D PBR Texture DiT, 1024px cascade)

### Why ConvRot INT8 (`TensorWiseINT8Layout`)?

3D DiT activations exhibit pronounced channel-wise outliers (especially in the `1536 -> 6144 -> 1536` FFN and cross-attention layers conditioned on DINOv3 patch tokens). Standard tensor-wise or naive row-wise INT8 quantization suffers severe clipping error on these channels.

`TensorWiseINT8Layout` with `convrot=True, convrot_groupsize=256, per_channel=True` applies an orthogonal block-diagonal **Walsh-Hadamard transform ($H_{256} / \sqrt{256}$)** along the reduction dimension $K$ in groups of 256:
$$(X H) (W H)^T = X (H H^T) W^T = X W^T$$
- **Offline (during quantization in `quantize_trellis2_convrot.py`)**: Each weight matrix $W \in \mathbb{R}^{N \times K}$ is rotated via group-256 Walsh-Hadamard transform and quantized per output channel (`scale` shape `[N]`, `float32`; `qdata` shape `[N, K]`, `int8`).
- **Online (during inference in `comfy-kitchen`)**: `comfy_kitchen.int8_linear` fuses the group-256 Walsh-Hadamard rotation and dynamic row-wise INT8 quantization of the input activation $X$ directly before the INT8 TensorCore GEMM.

### Layer Selection Table (Per 1.3B DiT Model)

| Layer Group | Count | Precision | Rationale |
| :--- | :---: | :---: | :--- |
| `blocks.{0..29}.self_attn.to_qkv` (`1536 -> 4608`) | 30 | **ConvRot INT8** | Compute-heavy projection; $K=1536$ is divisible by 256. |
| `blocks.{0..29}.self_attn.to_out` (`1536 -> 1536`) | 30 | **ConvRot INT8** | Compute-heavy projection; $K=1536$ is divisible by 256. |
| `blocks.{0..29}.cross_attn.to_q` (`1536 -> 1536`) | 30 | **ConvRot INT8** | Compute-heavy projection; $K=1536$ is divisible by 256. |
| `blocks.{0..29}.cross_attn.to_kv` (`1024 -> 3072`) | 30 | **ConvRot INT8** | Compute-heavy DINOv3 conditioning projection; $K=1024$ is divisible by 256. |
| `blocks.{0..29}.cross_attn.to_out` (`1536 -> 1536`) | 30 | **ConvRot INT8** | Compute-heavy projection; $K=1536$ is divisible by 256. |
| `blocks.{0..29}.mlp.mlp.0` (`1536 -> 6144`) | 30 | **ConvRot INT8** | FFN up-projection; $K=1536$ is divisible by 256. |
| `blocks.{0..29}.mlp.mlp.2` (`6144 -> 1536`) | 30 | **ConvRot INT8** | FFN down-projection; $K=6144$ is divisible by 256; fused with `gelu_tanh` + residual. |
| **Total Quantized Layers** | **210** | **`int8_tensorwise` (`convrot=True`)** | **99.1% of total parameter bytes** (`2464.9 MB -> 1252.4 MB`, **1.97×** reduction). |
| `input_layer`, `out_layer`, `t_embedder.mlp.{0,2}`, `adaLN_modulation`, `modulation`, `q_rms_norm`, `k_rms_norm` | **35** | **`bfloat16`** | Small parameter footprint (<0.9% of model size) with high sensitivity to quantization noise. |

Each quantized layer stores three keys in the `.safetensors` state dict:
- `<layer>.weight` (`torch.int8`, shape `[N, K]`)
- `<layer>.weight_scale` (`torch.float32`, shape `[N]`)
- `<layer>.comfy_quant` (`torch.uint8` JSON tensor: `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`)

---

## 3. `comfy-kitchen` Kernel Integrations

### 3.1 Mixed-Precision Quantized `SparseLinear` (`comfy/ops_sparse.py`)

In TRELLIS.2's `SLatFlowModel` (shape and texture stages), 3D features are stored as packed sparse voxel tensors (`VarLenTensor` / `SparseTensor`, where `input.feats` has shape `[N_active_voxels, C]`).
Originally, `comfy/ops_sparse.py` only defined `manual_cast.SparseLinear`, which could not load `comfy_quant` metadata or `QuantizedTensor` weights.

We added:
- `mixed_precision_sparse_ops(quant_config=..., compute_dtype=..., full_precision_mm=..., disabled=...)`
- `pick_sparse_operations(weight_dtype, compute_dtype, load_device=..., disable_fast_fp8=..., fp8_optimizations=..., model_config=...)`

In `mixed_precision_sparse_ops`, `SparseLinear` inherits directly from `MixedPrecisionOps(manual_cast).Linear` and delegates `forward(self, input)` to `super().forward(input.feats)` wrapped via `input.replace(...)`.

### 3.2 Fused `linear_input_act` (`gelu_tanh` + ConvRot + INT8 GEMM + Gated Residual)

In every transformer block (`ModulatedTransformerCrossBlock` and `ModulatedSparseTransformerCrossBlock`), the FFN sub-layer computes:
$$x \leftarrow x + \text{gate\_mlp} \odot W_2 \left(\text{GELU}_{\tanh}(W_1 \tilde{x} + b_1)\right) + b_2$$

By updating `FeedForwardNet` and `SparseFeedForwardNet` (`nodes/trellis2/model.py` and `comfy/ldm/trellis2/model.py`) to call:
```python
comfy.ops.linear_input_act(
    self.mlp[2],
    h,
    "gelu_tanh",
    residual=residual,
    residual_scale=residual_scale,
)
```
`TensorWiseINT8Layout.linear_input_act` dispatches to `comfy_kitchen.int8_linear` with:
- `input_act="gelu_tanh"`: Fuses approximate tanh-GELU activation with group-256 Walsh-Hadamard rotation and dynamic row-wise INT8 quantization in a single CUDA kernel pass over `h` (`[N, 6144]`).
- `residual=x, residual_scale=gate_mlp.reshape(-1)`: Fuses the gated residual epilogue (`x + gate_mlp * out`) directly into the CUTLASS INT8 GEMM dequantization epilogue when `batch_size == 1`.

### 3.3 Fused `comfy_kitchen.adaln` (60 Calls / Step)

Each of the 30 transformer blocks applies two Adaptive LayerNorm modulations (`norm1` before self-attention and `norm2` before FFN) with `elementwise_affine=False`:
$$\text{AdaLN}(x, \text{scale}, \text{shift}) = \text{LayerNorm}(x) \odot (1 + \text{scale}) + \text{shift}$$
`_fused_adaln` routes both dense `(B, L, C)` and sparse `(N_vox, C)` tensors directly to `comfy.quant_ops.ck.adaln(x, scale, shift, eps)`, eliminating intermediate tensor allocations across all 60 calls per denoising step.

### 3.4 Fused 3D `comfy_kitchen.apply_rope` (30 Calls / Step)

Both `RotaryPositionEmbedder` (dense 3D grid) and `SparseRotaryPositionEmbedder` (sparse 3D voxel coordinates) convert complex phases $e^{i\theta}$ to stacked $2\times 2$ rotation matrices $\begin{bmatrix}\cos\theta & \sin\theta \\ -\sin\theta & \cos\theta\end{bmatrix}$ via `_phases_to_freqs_cis` and invoke `comfy.ldm.flux.math.apply_rope` (`comfy_kitchen.apply_rope`) on $(q, k)$ in a single fused kernel call per block.

### 3.5 Zero-Copy `VarLenTensor` Broadcast for `batch_size == 1` (`comfy/sparse.py`)

Previously, `VarLenTensor.__elemwise__` always indexed `other = other[self.batch_boardcast_map]` whenever `other` was a `(B, C)` tensor, materializing a full `(N_vox, C)` tensor (`8192 × 1536` or `30000 × 1536`) six times per block (**180 allocations per step**) even when `batch_size == 1`.
We added a fast path in `comfy/sparse.py`:
```python
if len(self.layout) == 1 and other.shape[0] == 1:
    return self.replace(func(self.feats, other))
```
which relies on native PyTorch `(N_vox, C)` vs `(1, C)` broadcasting with zero memory allocation.

---

## 4. Verification & Reproducibility

Run the automated verification suite at any time:
```bash
python verify_convrot_trellis2.py
```

The script verifies:
1. **Layer Residency Audit**: Confirms `210` layers are loaded as `QuantizedTensor` (`TensorWiseINT8Layout`, `convrot=True`, `convrot_groupsize=256`, `torch.int8`) and `35` layers remain `torch.bfloat16`.
2. **Kernel Dispatch Audit**: Intercepts `comfy_kitchen` dispatch tables during a forward step and asserts:
   - `int8_linear == 210` (`180` via `F.linear` + `30` via fused `linear_input_act`)
   - `int8_linear_fused_act == 30`
   - `adaln == 60`
   - `apply_rope == 30`
3. **Numerical Parity**: Compares single-step DiT velocity predictions against the `bfloat16` baseline across `sparse_structure_flow_model`, `shape_slat_flow_model_512`, and `tex_slat_flow_model_512`, asserting cosine similarity $> 0.99$.
4. **VRAM & Latency**: Measures resident weight VRAM, peak forward VRAM, and synchronized CUDA latency (`ms/step`).
