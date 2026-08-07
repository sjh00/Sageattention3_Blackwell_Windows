# SageAttention3 Blackwell for Windows

**中文版文档：[README_zh.md](README_zh.md)**（English below）

Windows-oriented build of **SageAttention3** (FP4 microscaling attention for NVIDIA **Blackwell** GPUs: RTX 50-series / `sm_120`, and related `sm_100` / `sm_121`).

This repository packages the SageAttention3 Blackwell sources with **MSVC / Windows runtime fixes** so the wheel not only **builds**, but also **runs** correctly (avoids the common `CUDA error: misaligned address` crash).

> Upstream project: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)  
> Paper: [SageAttention3: Microscaling FP4 Attention](https://arxiv.org/abs/2505.11594)  
> Chinese docs: [README_zh.md](README_zh.md)

---

## Why this fork exists

Official SageAttention3 is Linux-first. On Windows + MSVC people often get:

1. **Compile errors** (`small` macro from Windows headers, ambiguous `std`, C++ standard flags).
2. **Silent runtime failures** after a successful wheel build:
   - `CUDA error: misaligned address`
   - Failures inside `sageattn3_blackwell` / `fp4attn_cuda.fwd`
3. **Triton dependency issues** on ComfyUI portable Python (missing `Python.h`), which breaks `per_block_mean=True`.

This tree applies practical Windows fixes (see [Windows fixes](#windows-fixes) below), verified on:

| Item | Example working setup |
|------|------------------------|
| OS | Windows 11 |
| GPU | RTX 50-series (compute capability 12.0) |
| Python | 3.13 |
| PyTorch | 2.13 + CUDA 13.2 |
| CUDA Toolkit | 13.2 |
| MSVC | VS 2022 Build Tools 14.44 |
| App | ComfyUI + KJNodes `PatchSageAttention` |

---

## Requirements

### Hardware

- NVIDIA **Blackwell** GPU only:
  - `sm_100` (B100-class)
  - `sm_120` (GeForce RTX 50-series)
  - `sm_121`

### Software

- **Python** 3.10+ (3.12 / 3.13 recommended; must match your app / ComfyUI embed)
- **PyTorch** with CUDA support, ideally **torch >= 2.8** and CUDA **12.8+**
- **CUDA Toolkit >= 12.8** (13.x recommended for 50-series), with `nvcc` on PATH or set via `CUDA_HOME`
- **Visual Studio 2022** Build Tools with **Desktop development with C++**
- **Git** (first build clones [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass))
- **ninja**, **packaging**, **wheel**, **build** (installed by the build script)

---

## Quick start (build wheel)

Open a normal `cmd` window in this folder (the script loads MSVC itself):

```bat
build_wheel.bat
```

Useful options:

```bat
rem Clean previous artifacts then build
build_wheel.bat --clean

rem Use ComfyUI portable Python
build_wheel.bat --python "D:\ComfyUI\python_embeded\python.exe"

rem Point at a specific CUDA Toolkit
build_wheel.bat --cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

rem Limit parallel compile jobs (default 2)
build_wheel.bat --jobs 1
```

On success, wheels appear under `dist\`:

```text
dist\sageattn3-1.0.0-cpXXX-cpXXX-win_amd64.whl
```

Full compiler log is written to `build.log`.

### Manual build (advanced)

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_HOME%\bin;%PATH%
python -m pip install -U pip setuptools wheel ninja packaging build
python -m build --wheel --no-isolation
```

---

## Install the wheel

Match the **same Python** you used to build (ABI tag must match, e.g. `cp313`):

```bat
python -m pip install --force-reinstall --no-deps dist\sageattn3-1.0.0-cp313-cp313-win_amd64.whl
```

ComfyUI portable example:

```bat
D:\ComfyUI\python_embeded\python.exe -m pip install --force-reinstall --no-deps dist\sageattn3-1.0.0-cp313-cp313-win_amd64.whl
```

Then **restart ComfyUI**.

### Smoke test

```bat
python -c "import torch; from sageattn3 import sageattn3_blackwell; q=torch.randn(1,8,128,128,device='cuda',dtype=torch.bfloat16); print(sageattn3_blackwell(q,q,q,per_block_mean=False).shape)"
```

---

## Usage

```python
from sageattn3 import sageattn3_blackwell

# q, k, v: FP16 or BF16, shape (batch, heads, seq_len, head_dim)
# head_dim must be 64 or 128 (values >= 256 fall back to SDPA)
out = sageattn3_blackwell(
    q, k, v,
    is_causal=False,
    per_block_mean=False,  # True enables 128-token group mean centering
)
```

### ComfyUI (KJNodes)

1. Install the wheel into ComfyUI’s Python.
2. Restart ComfyUI.
3. Use **PatchSageAttentionKJ** with:
   - `sageattn3` → `per_block_mean=False`
   - `sageattn3_per_block_mean` → `per_block_mean=True` (no Triton required in this fork)

### Layout note

API expects **HND** layout: `(B, H, L, D)`.  
KJNodes converts from Comfy’s NHD when needed.

---

## Windows fixes

Compared with upstream SageAttention3 Blackwell, this project includes:

| Area | Fix |
|------|-----|
| MSVC kernel launch | Pass over-aligned kernel params **by pointer** via a device `DeviceParamsPack` (MSVC cannot pass `alignas(128)` / `CUTE_GRID_CONSTANT` by value) |
| TMA descriptors | `alignas(TMA_*)` on mainloop / epilogue TMA fields for correct `prefetch_tma_descriptor` |
| Scheduler | Use real `multiProcessorCount` instead of hard-coded `170` SMs |
| Compiler flags | C++20, `/Zc:__cplusplus`, `/Zc:preprocessor`, `/bigobj`, `-Usmall`, `USE_CUDA` |
| Headers | `#undef small` guards (Windows `rpcndr.h` defines `small` as `char`) |
| Preprocess | Pure **PyTorch** group mean (no Triton) so ComfyUI embed Python works with `per_block_mean` |

These draw on community work around upstream PRs/issues such as [#323](https://github.com/thu-ml/SageAttention/pull/323), [#355](https://github.com/thu-ml/SageAttention/pull/355), [#370](https://github.com/thu-ml/SageAttention/pull/370), and related discussions on misaligned-address failures.

---

## Limitations (same as upstream)

SageAttention3 works well for many **image** models and some **video** models (e.g. CogVideoX-2B, HunyuanVideo, Mochi). It is **not guaranteed lossless** for every model.

Tips:

- Prefer **head_dim 64 or 128**.
- If quality drops, mix SageAttention2++ on some layers/timesteps with SageAttention3 on others.
- Do **not** set `CUDA_LAUNCH_BLOCKING=1` while debugging SA3; it can disturb TMA timing.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `cl.exe` / `vcvars` not found | Install VS 2022 Build Tools + C++ workload; re-run `build_wheel.bat` |
| `nvcc` not found | Install CUDA Toolkit 12.8+; pass `--cuda "..."` |
| `Unsupported GPU` | Need Blackwell (`sm_100` / `sm_120` / `sm_121`) |
| Wheel installs but import fails | Rebuild with the **same** Python as the runtime (cp312 vs cp313) |
| `misaligned address` after custom edits | Keep MSVC pointer launch path and `/Zc:__cplusplus` |
| ComfyUI still errors after install | Fully restart ComfyUI; confirm `pip show sageattn3` path is under `python_embeded` |
| Build OOM / compiler crash | `build_wheel.bat --jobs 1` |

---

## Project layout

```text
sageattention3_blackwell_Windows/
  build_wheel.bat      # one-click Windows build
  setup.py             # CUDA extensions + MSVC flags
  sageattn3/           # Python API + CUDA sources
  LICENSE              # Apache-2.0 (upstream)
  README.md
  README_zh.md
```

First build clones CUTLASS into `csrc/cutlass/` (gitignored).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).  
Upstream copyright: SageAttention team / THU-ML.

---

## Citation

If you use SageAttention3, please cite the original authors:

```bibtex
@article{zhang2025sageattention3,
  title={SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training},
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Xu, Xiaoming and Huang, Haofeng and Wang, Haoxu and Jiang, Kai and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.11594},
  year={2025}
}
```

Also see SageAttention / SageAttention2 / SageAttention2++ citations in the [upstream README](https://github.com/thu-ml/SageAttention).

---

## Disclaimer

This is an **unofficial Windows packaging / portability patch set**, not an official THU-ML release.  
API package name remains `sageattn3` for drop-in use with existing tools (e.g. ComfyUI-KJNodes).
