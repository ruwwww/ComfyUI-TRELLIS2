---
license: mit
base_model:
  - microsoft/TRELLIS.2-4B
  - microsoft/TRELLIS-image-large
pipeline_tag: image-to-3d
library_name: comfyui
tags:
  - 3d
  - image-to-3d
  - pbr
  - trellis
  - trellis2
  - comfyui
  - comfy-kitchen
  - quantization
  - int8
  - convrot
---

# TRELLIS.2-4B — ConvRot INT8 (`comfy-kitchen`) Quantized Checkpoints for ComfyUI

This repository provides **ConvRot INT8 (`TensorWiseINT8Layout`, `convrot=True, convrot_groupsize=256`)** quantized checkpoints for all five **1.3B Diffusion Transformer (DiT)** flow models in Microsoft Research's [**TRELLIS.2-4B**](https://huggingface.co/microsoft/TRELLIS.2-4B) image-to-3D + PBR texturing pipeline, tailored for native hardware-accelerated execution via **`comfy-kitchen`** in [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) and [**`ruwwww/ComfyUI-TRELLIS2`**](https://github.com/ruwwww/ComfyUI-TRELLIS2) (forked from [**`PozzettiAndrea/ComfyUI-TRELLIS2`**](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2)).

- **1.97× smaller DiT checkpoints** (`~2.46 GB` → `~1.25 GB` per DiT model; `~12.3 GB` → `~6.26 GB` across all 5 DiTs)
- **1.51×–1.59× faster DiT forward pass** when paired with `comfy-kitchen` (`int8_linear`, fused `gelu_tanh` + gated residual epilogue, fused `adaln`, and fused 3D `apply_rope`)
- **Near-lossless quality** (`0.9981–0.9994` cosine similarity and `24.1–29.1 dB` SNR vs. `bfloat16` reference)

---

## Credits, Attribution & Fair Acknowledgments

This model distribution is a quantized derivative of upstream open-source research and engineering. Please credit and cite the upstream creators:

1. **Original Model & Research — Microsoft Research ([`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) & [`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large))**
   - *Authors*: Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, Jiaolong Yang (Microsoft Research Asia).
   - Released under the **MIT License**.
2. **ComfyUI-TRELLIS2 Custom Node & Pipeline Packaging — [Andrea Pozzetti (`@PozzettiAndrea`)](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2)**
   - **Andrea Pozzetti ([@PozzettiAndrea](https://github.com/PozzettiAndrea))** created [`ComfyUI-TRELLIS2`](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2), the progressive stage-loading architecture, O-Voxel integration, and Comfy3D workflow ecosystem, with contributions from **[@Fristender](https://github.com/Fristender)**.
3. **`comfy-kitchen` Kernels & Native ComfyUI TRELLIS.2 Support — [ComfyOrg / ComfyUI](https://github.com/comfyanonymous/ComfyUI)**
   - **[@comfyanonymous](https://github.com/comfyanonymous)**: `comfy-kitchen` (`int8_linear`, `adaln`, `apply_rope`, `TensorWiseINT8Layout`, and group-wise Walsh-Hadamard `ConvRot` rotation kernels) and `MixedPrecisionOps`.
   - **Jukka Seppänen ([@kijai](https://github.com/kijai))**: Native ComfyUI TRELLIS.2 (`comfy/ldm/trellis2`), sparse tensor infrastructure (`comfy/sparse.py`, `comfy/ops_sparse.py`), and `int8_convrot` support.
   - **[@rattus128](https://github.com/rattus128)**: `comfy.ops.linear_input_act` prologue/epilogue fusion API.
   - **Jialong (Bruce) Li ([@chelsealong](https://github.com/chelsealong))**, **[@liminfei-amd](https://github.com/liminfei-amd)**, **Tijana Vukovic ([@tvukovic-amd](https://github.com/tvukovic-amd))**, and **[@DELUXA](https://github.com/DELUXA)**: `comfy.quant_ops` and cross-platform `comfy-kitchen` kernel support.

---

## Included Files (`ckpts/`)

| Checkpoint File | Stage / Role | Quantized Layers | Kept in `BF16` | Original `BF16` Size | `INT8 ConvRot` Size | Compression |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `ckpts/ss_flow_img_dit_1_3B_64_int8_convrot.safetensors` (+ `.json`) | Stage 1: Sparse Structure Flow DiT ($64^3$) | 210 | 35 | 2,464.6 MB | **1,252.2 MB** | **1.97×** |
| `ckpts/slat_flow_img2shape_dit_1_3B_512_int8_convrot.safetensors` (+ `.json`) | Stage 2: Shape SLat Flow DiT ($512^3$) | 210 | 35 | 2,464.8 MB | **1,252.3 MB** | **1.97×** |
| `ckpts/slat_flow_img2shape_dit_1_3B_1024_int8_convrot.safetensors` (+ `.json`) | Stage 2 Cascade: Shape SLat Flow DiT ($1024^3$) | 210 | 35 | 2,464.8 MB | **1,252.3 MB** | **1.97×** |
| `ckpts/slat_flow_imgshape2tex_dit_1_3B_512_int8_convrot.safetensors` (+ `.json`) | Stage 3: PBR Texture SLat Flow DiT ($512^3$) | 210 | 35 | 2,464.9 MB | **1,252.4 MB** | **1.97×** |
| `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_int8_convrot.safetensors` (+ `.json`) | Stage 3 Cascade: PBR Texture SLat Flow DiT ($1024^3$) | 210 | 35 | 2,464.9 MB | **1,252.4 MB** | **1.97×** |

*(Note: The lightweight decoders `ss_dec_conv3d_16l8_fp16`, `shape_dec_next_dc_f16c32_fp16`, `tex_dec_next_dc_f16c32_fp16`, and `shape_enc_next_dc_f16c32_fp16` remain in their original `fp16` format from `microsoft/TRELLIS.2-4B` / `microsoft/TRELLIS-image-large`).*

---

## Quantization Methodology (`int8_tensorwise` + `ConvRot`)

Each 1.3B DiT consists of 30 modulated transformer blocks (`channels=1536, num_heads=12, head_dim=128, ffn=6144`):

- **Quantized to ConvRot INT8 (`210` layers / model)**:
  - Self-attention projections: `blocks.{0..29}.self_attn.{to_qkv, to_out}`
  - Cross-attention projections: `blocks.{0..29}.cross_attn.{to_q, to_kv, to_out}`
  - Feed-Forward Network projections: `blocks.{0..29}.mlp.mlp.{0, 2}`
  - Uses `comfy_kitchen.tensor.TensorWiseINT8Layout` with **group-256 Walsh-Hadamard rotation (`convrot=True, convrot_groupsize=256, per_channel=True`)**. Rotating weights and activations in 256-element blocks disperses channel-wise activation outliers prior to dynamic INT8 quantization, preserving high SNR without calibration data.
- **Preserved in `bfloat16` (`35` tensors / model)**:
  - `input_layer`, `out_layer`, `t_embedder.mlp.{0, 2}`, `adaLN_modulation`, block `modulation` parameters, and `q_rms_norm` / `k_rms_norm` scales (<0.9% of total bytes).

---

## Benchmarks & Numerical Quality

Evaluated against the official `bfloat16` checkpoints on deterministic 3D dense ($16^3$) and sparse ($8,192$ active voxels) inputs:

| Model Stage | Weight VRAM (`BF16` → `INT8`) | Peak VRAM (`BF16` → `INT8`) | Latency / Step (`BF16` → `INT8`) | Speedup | Cosine Similarity | Relative L2 Error | SNR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sparse_structure_flow_model`** | `2482.1 MB` → **`1262.9 MB`** | `2692.3 MB` → **`1409.1 MB`** | `367.04 ms` → **`231.44 ms`** | **1.59×** | **0.998709** | `0.0509` | **25.86 dB** |
| **`shape_slat_flow_model_512`** | `2480.3 MB` → **`1260.5 MB`** | `2867.4 MB` → **`1608.2 MB`** | `774.00 ms` → **`511.12 ms`** | **1.51×** | **0.999379** | `0.0352` | **29.06 dB** |
| **`tex_slat_flow_model_512`** | `2480.4 MB` → **`1260.6 MB`** | `2869.2 MB` → **`1609.9 MB`** | `773.32 ms` → **`510.87 ms`** | **1.51×** | **0.998068** | `0.0622` | **24.13 dB** |

---

## Usage in ComfyUI

1. Place `pipeline.json` in `ComfyUI/models/trellis2/` and the `*_int8_convrot.safetensors` + `.json` files in `ComfyUI/models/trellis2/ckpts/`.
2. Ensure your ComfyUI environment has `comfy-kitchen >= 0.2.36` installed.
3. The updated `ComfyUI-TRELLIS2` loader (`nodes/stages.py`) automatically detects and loads `*_int8_convrot.safetensors` with `comfy.ops.mixed_precision_ops` and `comfy.ops_sparse.mixed_precision_sparse_ops`.

---

## Citation

If you use these models in your research or production pipeline, please cite the original **TRELLIS** / **TRELLIS.2** work by Microsoft Research:

```bibtex
@article{xiang2024structured,
  title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
  author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
  journal = {arXiv preprint arXiv:2412.01506},
  year    = {2024}
}
```

## License

Released under the [MIT License](https://opensource.org/licenses/MIT), matching the upstream license of [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) and [`PozzettiAndrea/ComfyUI-TRELLIS2`](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2).
