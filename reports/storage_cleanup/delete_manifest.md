# Storage Cleanup Delete Manifest

Generated: `2026-07-20T12:51:53.921658+09:00`

Validated exact targets: **784**
Expected reclaimable allocation: **84.299 GB**

Every JSON target is `VERIFIED_DISPOSABLE`. The executor must recheck identity, workers, handles,
resolved containment, overlap-window writes, and selected/protected exclusions immediately before deletion.

| Category | Exact targets | Estimated GB |
|---|---:|---:|
| VLA WSL crash dump | 2 | 39.943 |
| closed-route public OpenVLA-OFT checkpoint | 1 | 15.941 |
| uv package/source cache | 10 | 8.582 |
| pip package-download cache | 7 | 7.129 |
| Conda package-download/extraction cache | 748 | 4.253 |
| closed-route clean public source clone | 2 | 3.117 |
| nonselected public base-model cache | 1 | 2.035 |
| nonselected public Hugging Face model cache | 1 | 2.035 |
| closed RL4IL prior feature cache | 1 | 1.214 |
| Torch Hub cache | 2 | 0.047 |
| Hugging Face transfer log | 5 | 0.004 |
| nonselected LightVLA metadata cache | 1 | 0.000 |
| stale Hugging Face lock cache | 3 | 0.000 |

## Largest exact targets

| Platform | Exact path | GB | Evidence |
|---|---|---:|---|
| windows | `C:\Users\jiheo\AppData\Local\Temp\wsl-crashes\wsl-crash-1784097862-307-_home_jiheon_miniconda3-official_envs_official-smolvla-libero_bin_python3.10-6.dmp` | 20.145 | `9D17164F0C822CC56D9FEDB32155FFB94A4F683B624CDFE2BA6B229310AE28AF` |
| windows | `C:\Users\jiheo\AppData\Local\Temp\wsl-crashes\wsl-crash-1784097610-314-_home_jiheon_miniconda3-official_envs_official-smolvla-libero_bin_python3.10-6.dmp` | 19.798 | `8F3A0CA82FA76E07CB17F1A8CA40C57B00DC8CB85E02CBFE26CFF67820FF0198` |
| wsl | `/home/jiheon/assets/checkpoints/openvla-oft/moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | 15.941 | `638918f3d1c2e43a39a8a20772bdb8b91835e4b7` |
| wsl | `/home/jiheon/.cache/uv/archive-v0` | 8.339 | `enumerated cache` |
| windows | `C:\Users\jiheo\AppData\Local\pip\cache\http-v2` | 4.125 | `enumerated cache` |
| wsl | `/home/jiheon/.cache/pip/http-v2` | 2.984 | `enumerated cache` |
| wsl | `/home/jiheon/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct` | 2.035 | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| windows | `C:\Users\jiheo\.cache\huggingface\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct` | 2.035 | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| windows | `C:\assets\repos\PCD` | 1.723 | `cec18b820daeadfdaf080c030a1b5eb080ff75cd` |
| windows | `C:\assets\repos\VLA-Arena` | 1.394 | `babe582ebffc82b979b77964a7e56417d02f63a4` |
| wsl | `/home/jiheon/.cache/huggingface/hub/models--openai--clip-vit-base-patch32` | 1.214 | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| wsl | `/home/jiheon/miniconda3-official/pkgs/cache` | 0.615 | `enumerated cache` |
| wsl | `/home/jiheon/miniconda3-official/pkgs/sysroot_linux-64-2.34-h087de78_3` | 0.482 | `enumerated cache` |
| wsl | `/home/jiheon/miniconda3-official/pkgs/gcc_impl_linux-64-15.2.0-he0086c7_19` | 0.277 | `enumerated cache` |
| wsl | `/home/jiheon/.cache/uv/git-v0` | 0.237 | `enumerated cache` |
| windows | `C:\Users\jiheo\miniconda3\pkgs\cache` | 0.124 | `enumerated cache` |
| wsl | `/home/jiheon/miniconda3-official/pkgs/python-3.11.15-h7508c33_1_cpython` | 0.107 | `enumerated cache` |
| wsl | `/home/jiheon/miniconda3-official/pkgs/icu-78.3-h53478e7_1` | 0.098 | `enumerated cache` |
| wsl | `/home/jiheon/miniconda3-official/pkgs/cmake-4.1.3-hc85cc9f_0` | 0.095 | `enumerated cache` |
| windows | `C:\Users\jiheo\miniconda3\pkgs\libmamba-2.3.2-hc213065_1` | 0.087 | `enumerated cache` |

The complete exact target list and hashes are in `delete_manifest.json`.
