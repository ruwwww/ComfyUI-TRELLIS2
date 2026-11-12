> [!WARNING]
> Warning: Upstream installation uses the experimental package `comfy-env` to attempt a one-click isolated install, which downloads and uses the `pixi` package manager.

# ComfyUI-TRELLIS2 (ConvRot INT8 + `comfy-kitchen` Accelerated Fork)

ComfyUI custom nodes for [**TRELLIS.2**](https://github.com/microsoft/TRELLIS.2) — Microsoft Research's state-of-the-art image-to-3D generation model for high-quality 3D meshes with Physically Based Rendering (PBR) materials from a single image.

This fork builds directly upon [**Andrea Pozzetti's (`@PozzettiAndrea`) `ComfyUI-TRELLIS2`**](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2) and integrates **ComfyUI's native `comfy-kitchen` CUDA/Triton kernels** and **ConvRot INT8 (`TensorWiseINT8Layout`) quantization** across all 5 TRELLIS.2 1.3B DiT flow models (Structure, Shape 512/1024, and Texture 512/1024), cutting DiT weight VRAM in half (**1.97× smaller**) and accelerating per-step inference by **1.51×–1.59×** while preserving near-lossless numerical fidelity (**24.1–29.1 dB SNR**, **0.9981–0.9994 cosine similarity**).

<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-TRELLIS2/">
<img src="https://pozzettiandrea.github.io/ComfyUI-TRELLIS2/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-TRELLIS2/">View Upstream Live Test Gallery →</a></b>
</div>

---

## Credits, Lineage & Acknowledgments

This work stands on the shoulders of the original researchers, upstream node authors, and kernel engineers. Full credit goes to:

1. **Microsoft Research — [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) & [TRELLIS](https://github.com/microsoft/TRELLIS)**
   - *Authors*: Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, Jiaolong Yang, and the Microsoft Research Asia team.
   - Base checkpoints: [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) and [`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large).
2. **Upstream Custom Node — [`PozzettiAndrea/ComfyUI-TRELLIS2`](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2)**
   - *Lead Author & Maintainer*: **Andrea Pozzetti ([@PozzettiAndrea](https://github.com/PozzettiAndrea))** — designed and built the `ComfyUI-TRELLIS2` node suite, progressive stage loading (`nodes/stages.py`), native ComfyUI `ModelPatcher` / `KSampler` integration, O-Voxel dual-grid mesh conditioning, and the [`Comfy3D`](https://discord.gg/bcdQCUjnHE) ecosystem.
   - *Upstream Contributors*: **[@Fristender](https://github.com/Fristender)** and the Comfy3D community.
3. **ComfyUI Core & `comfy-kitchen` Kernel Infrastructure — [ComfyOrg / ComfyUI](https://github.com/comfyanonymous/ComfyUI)**
   - **[@comfyanonymous](https://github.com/comfyanonymous)** — `comfy-kitchen` (`int8_linear`, `adaln`, `apply_rope`, `TensorWiseINT8Layout`, group-wise Walsh-Hadamard `ConvRot` rotation kernels), `MixedPrecisionOps`, and TRELLIS.2 VRAM/RAM optimizations.
   - **Jukka Seppänen ([@kijai](https://github.com/kijai))** — ComfyUI native TRELLIS.2 (`comfy/ldm/trellis2`), sparse tensor infrastructure (`comfy/sparse.py`, `comfy/ops_sparse.py`), `o-voxel` remeshing integration, and `int8_convrot` format support.
   - **[@rattus128](https://github.com/rattus128)** — `comfy.ops.linear_input_act` prologue/epilogue fusion API (fusing activation + `ConvRot` + INT8 GEMM + gated residual epilogue).
   - **Jialong (Bruce) Li ([@chelsealong](https://github.com/chelsealong))**, **[@liminfei-amd](https://github.com/liminfei-amd)**, **Tijana Vukovic ([@tvukovic-amd](https://github.com/tvukovic-amd))**, and **[@DELUXA](https://github.com/DELUXA)** — `comfy.quant_ops` and cross-platform `comfy-kitchen` dispatch enhancements.

---

## What This Fork Adds

| Feature | Description |
| :--- | :--- |
| **ConvRot INT8 DiT Quantization (`TensorWiseINT8Layout`)** | Quantizes all **210** transformer block `Linear` and `SparseLinear` layers across all 5 TRELLIS.2 1.3B DiT models using group-256 Walsh-Hadamard rotation (`convrot=True, convrot_groupsize=256, per_channel=True`) to suppress activation/weight outliers prior to INT8 quantization, while preserving the **35** sensitive stem, timestep, AdaLN modulation, and QK-RMSNorm tensors in `bfloat16`. |
| **Mixed-Precision Quantized `SparseLinear`** | Extends `comfy.ops_sparse` (`mixed_precision_sparse_ops` / `pick_sparse_operations`) so sparse 3D `VarLenTensor` features (`SLatFlowModel` shape & texture) natively load `QuantizedTensor` weights and dispatch to `comfy_kitchen.int8_linear`. |
| **Fused `linear_input_act` (`gelu_tanh` + Gated Residual)** | Routes `FeedForwardNet.mlp[2]` and `SparseFeedForwardNet.mlp[2]` through `comfy.ops.linear_input_act(..., "gelu_tanh", residual=x, residual_scale=gate_mlp)`, fusing `GELU(tanh)`, Walsh-Hadamard rotation, dynamic row-wise INT8 input quantization, INT8 TensorCore GEMM, and gated residual addition into a single kernel call (**30 calls/step**). |
| **Fused `comfy_kitchen.adaln` & 3D `apply_rope`** | Replaces elementwise LayerNorm + scale/shift modulation with `comfy_kitchen.adaln` (**60 calls/step**) and routes dense & sparse 3D Rotary Position Embeddings through `comfy_kitchen.apply_rope` (**30 calls/step**). |
| **Zero-Copy `VarLenTensor` Batch=1 Broadcast** | Eliminates redundant `other[self.batch_boardcast_map]` index-gather allocations (`180` full `(N, C)` tensor allocations per step) when `batch_size == 1`. |

For full architectural details, see [**`docs/CONVROT_INT8_KITCHEN.md`**](docs/CONVROT_INT8_KITCHEN.md).

---

## Benchmarks & Numerical Parity

Measured via [`verify_convrot_trellis2.py`](verify_convrot_trellis2.py) against the original `bfloat16` checkpoints on deterministic 3D dense (`16³`) and sparse (`8192` active voxels) inputs:

| Stage / Model | Weight VRAM (`BF16` → `INT8 ConvRot`) | Peak VRAM (`BF16` → `INT8 ConvRot`) | Step Latency (`BF16` → `INT8 ConvRot`) | Active `comfy-kitchen` Kernels / Step | Cosine Sim | Rel L2 | SNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`sparse_structure_flow_model`**<br>*(Dense 3D DiT, 1.3B)* | `2482.1 MB` → **`1262.9 MB`** (**1.97×**) | `2692.3 MB` → **`1409.1 MB`** | `367.04 ms` → **`231.44 ms`** (**1.59×**) | `int8_linear`: **210** *(30 fused)*<br>`adaln`: **60**, `apply_rope`: **30** | **0.998709** | `0.0509` | **25.86 dB** |
| **`shape_slat_flow_model_512`**<br>*(Sparse 3D DiT, 1.3B)* | `2480.3 MB` → **`1260.5 MB`** (**1.97×**) | `2867.4 MB` → **`1608.2 MB`** | `774.00 ms` → **`511.12 ms`** (**1.51×**) | `int8_linear`: **210** *(30 fused)*<br>`adaln`: **60**, `apply_rope`: **30** | **0.999379** | `0.0352` | **29.06 dB** |
| **`tex_slat_flow_model_512`**<br>*(Sparse Texture DiT, 1.3B)* | `2480.4 MB` → **`1260.6 MB`** (**1.97×**) | `2869.2 MB` → **`1609.9 MB`** | `773.32 ms` → **`510.87 ms`** (**1.51×**) | `int8_linear`: **210** *(30 fused)*<br>`adaln`: **60**, `apply_rope`: **30** | **0.998068** | `0.0622` | **24.13 dB** |

---

## Installation & Usage

### 1. Install the Custom Node

1. **ComfyUI Manager (Upstream)** — search for `TRELLIS2` in ComfyUI Manager, or install via Git URL.
2. **Manual Installation**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ruwwww/ComfyUI-TRELLIS2.git
   cd ComfyUI-TRELLIS2
   pip install -r requirements.txt --upgrade
   python install.py
   ```
   *(Ensure `comfy-kitchen >= 0.2.36` is installed in your ComfyUI Python environment).*

### 2. Using ConvRot INT8 Checkpoints

Download the quantized checkpoints from [**`ruwwww/TRELLIS.2-4B-int8-convrot`**](https://huggingface.co/ruwwww/TRELLIS.2-4B-int8-convrot) and place the `*_int8_convrot.safetensors` and `*_int8_convrot.json` files inside `ComfyUI/models/trellis2/ckpts/`:

- `ss_flow_img_dit_1_3B_64_int8_convrot.safetensors` (+ `.json`)
- `slat_flow_img2shape_dit_1_3B_512_int8_convrot.safetensors` (+ `.json`)
- `slat_flow_img2shape_dit_1_3B_1024_int8_convrot.safetensors` (+ `.json`)
- `slat_flow_imgshape2tex_dit_1_3B_512_int8_convrot.safetensors` (+ `.json`)
- `slat_flow_imgshape2tex_dit_1_3B_1024_int8_convrot.safetensors` (+ `.json`)

`nodes/stages.py` automatically detects and prefers `*_int8_convrot.safetensors` whenever present in `ComfyUI/models/trellis2/ckpts/`. To force the original `bfloat16` checkpoints for comparison without moving files, set:
```bash
export TRELLIS2_FORCE_BF16=1
```

### 3. Quantizing Checkpoints Yourself & Running Verification

To quantize local `bfloat16` TRELLIS.2 checkpoints to ConvRot INT8:
```bash
python quantize_trellis2_convrot.py
```

To run the full layer audit, `comfy-kitchen` op counter, numerical parity test, and latency/VRAM benchmark:
```bash
python verify_convrot_trellis2.py
```

---

## Example Workflows

![tpose](docs/tpose.png)

![rmbg](docs/rmbg.png)

https://github.com/user-attachments/assets/e28e4a74-b119-4303-8e30-63361f26aa88

## Community

- Upstream issues & discussions: [PozzettiAndrea/ComfyUI-TRELLIS2 Discussions](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2/discussions)
- Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## License

MIT License — see [LICENSE](LICENSE). Original TRELLIS.2 weights and code are released by Microsoft Research under the MIT License.
