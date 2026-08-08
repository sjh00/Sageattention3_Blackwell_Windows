# SageAttention3 Blackwell Windows 版

**英文版文档：[README.md](README.md)**（中文内容如下）

面向 **Windows** 的 **SageAttention3**（NVIDIA **Blackwell** GPU：RTX 50 系 / `sm_120`，以及 `sm_100` / `sm_121`）构建工程。

本仓库在官方 SageAttention3 Blackwell 源码基础上，加入 **MSVC / Windows 运行时修复**，使生成的 wheel **既能干净编译，又能稳定运行**（避免常见的 `CUDA error: misaligned address` 崩溃）。

> 上游项目：[thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)  
> 论文：[SageAttention3: Microscaling FP4 Attention](https://arxiv.org/abs/2505.11594)  
> English docs: [README.md](README.md)

---

## 为什么需要这个仓库

官方 SageAttention3 以 Linux 为主。在 Windows + MSVC 上常见问题：

1. **编译失败**（Windows 头文件里的 `small` 宏、`std` 歧义、C++ 标准开关等）。
2. **编译过程“半成功”**（标志冲突、无穷大量告警、Python ABI 混用等，可能影响结果稳定性）。
3. **wheel 能装却一用就崩**：
   - `CUDA error: misaligned address`
   - 栈落在 `sageattn3_blackwell` / `fp4attn_cuda.fwd`
4. **Triton 在 ComfyUI 便携 Python 上不可用**（缺少 `Python.h`），导致 `per_block_mean=True` 失败。

本仓库提供一套已验证的 Windows 补丁（见 [Windows 修复说明](#windows-修复说明)）。参考成功环境：

| 项目 | 示例 |
|------|------|
| 系统 | Windows 11 |
| 显卡 | RTX 50 系（compute capability 12.0） |
| Python | 3.13（亦支持 3.14 / free-threaded `3.14t`，可用 uv 部署） |
| PyTorch | 2.13 + CUDA 13.2 |
| CUDA Toolkit | 13.2 |
| MSVC | **Visual Studio 2026** Build Tools（MSVC 14.5x；安装目录可能为 `...\18\BuildTools`） |
| 应用 | ComfyUI + KJNodes `PatchSageAttention` |

---

## 环境要求

### 硬件

- 仅支持 NVIDIA **Blackwell**：
  - `sm_100`
  - `sm_120`（GeForce RTX 50 系）
  - `sm_121`

### 软件

- **Python** 3.10+（推荐 3.12 / **3.13**；亦支持 **3.14 / 3.14t free-threaded**；**构建与运行 ABI 必须一致**）
- **带 CUDA 的 PyTorch**（建议 torch >= 2.8，CUDA 12.8+）
- **CUDA Toolkit >= 12.8**（50 系建议 13.x），需有 `nvcc`，或通过 `CUDA_HOME` / `--cuda` 指定
- **Visual Studio 2026** Build Tools，勾选 **使用 C++ 的桌面开发**  
  （`build_wheel.bat` 若本机仍装有 VS 2022 也可探测使用；**当前以 2026 为主要验证环境**）
- **Git**（首次构建会把 [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) 克隆到 `csrc/cutlass/`）
- **ninja / packaging / wheel / build**（`build_wheel.bat` 通过 **`uv pip`** 安装；无 uv 时才回退 `python -m pip`）
- 可选：**[uv](https://github.com/astral-sh/uv)**（推荐，用于管理 Python 与依赖）

---

## 快速开始：编译 wheel

在本目录打开 `cmd`（脚本会自行加载 MSVC 环境）：

```bat
build_wheel.bat
```

常用参数：

```bat
rem 清理旧产物再编（切换 3.13 / 3.14t 后强烈建议）
build_wheel.bat --clean

rem 项目 / uv venv 的 Python（相对路径即可）
build_wheel.bat --python ".venv\Scripts\python.exe"

rem ComfyUI 便携 Python（盘符与目录按本机安装修改）
build_wheel.bat --python "D:\ComfyUI\python_embeded\python.exe"

rem 指定 CUDA Toolkit 路径
build_wheel.bat --cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

rem 并行编译任务数（默认 2，内存不够可改 1）
build_wheel.bat --jobs 1
```

成功后 wheel 在 `dist\`：

```text
dist\sageattn3-1.0.0-cp313-cp313-win_amd64.whl
dist\sageattn3-1.0.0-cp314-cp314t-win_amd64.whl   （free-threaded 3.14t 示例）
```

完整编译日志：`build.log`。脚本还会扫描日志中的硬错误，以及历史上容易误导的告警类别（`D9025`、`#221`、`#68`）。

### 手动编译（进阶）

```bat
rem VS 2026 Build Tools（安装路径主版本号常为 "18"）
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_HOME%\bin;%PATH%

rem uv 环境请用 uv pip（venv 默认没有 pip 模块）
uv pip install --python .venv\Scripts\python.exe -U setuptools wheel ninja packaging build
.venv\Scripts\python.exe -m build --wheel --no-isolation
```

若 `vcvars64.bat` 在其他 edition（`Community` / `Professional` / `Enterprise`）下，请改对应路径。`build_wheel.bat` 会自动探测常见布局。

---

## 安装 wheel

**构建用的 Python 与运行环境必须一致**（ABI 标签要对上，例如 `cp313` 或 free-threaded 的 `cp314t`）：

```bat
rem 推荐：uv 环境（Python 3.13 示例）
uv pip install --python .venv\Scripts\python.exe --force-reinstall --no-deps dist\sageattn3-1.0.0-cp313-cp313-win_amd64.whl

rem free-threaded 3.14t 示例
uv pip install --python .venv\Scripts\python.exe --force-reinstall --no-deps dist\sageattn3-1.0.0-cp314-cp314t-win_amd64.whl

rem 传统 pip 环境（仅当本机确有 pip 时）
python -m pip install --force-reinstall --no-deps dist\sageattn3-1.0.0-cp313-cp313-win_amd64.whl
```

ComfyUI 便携包示例：

```bat
uv pip install --python "D:\ComfyUI\python_embeded\python.exe" --force-reinstall --no-deps dist\sageattn3-1.0.0-cp313-cp313-win_amd64.whl
```

安装后请 **完全重启 ComfyUI**。

> **3.14 free-threaded 说明**：加载扩展时可能出现 “GIL has been enabled…” 的 `RuntimeWarning`，这是预期行为（CUDA 扩展未声明无 GIL 安全，解释器会自动重新启用 GIL）。

### 冒烟测试

```bat
python -c "import torch; from sageattn3 import sageattn3_blackwell; q=torch.randn(1,8,128,128,device='cuda',dtype=torch.bfloat16); print(sageattn3_blackwell(q,q,q,per_block_mean=False).shape)"
```

与 torch SDPA 对比（质量 + 大 shape 速度 + 因果诊断）：

```bat
python scripts/smoke_vs_sdpa.py
python scripts/smoke_vs_sdpa.py --quick
python scripts/smoke_vs_sdpa.py --causal-diag
python scripts/smoke_vs_sdpa.py --bench-only
```

预期数值与因果结论见 [相对 SDPA 的质量说明](#相对-sdpa-的质量说明)。

---

## 相对 SDPA 的质量说明

在 Blackwell（`sm_120`、bf16、HND）上，将 `sageattn3_blackwell` 与 `torch.nn.functional.scaled_dot_product_attention` 对比的代表性结果如下。请在本机用 `scripts/smoke_vs_sdpa.py` 复测。

| 场景 | 相对 SDPA 余弦（典型） | 说明 |
|------|------------------------|------|
| 非因果（`is_causal=False`） | **~0.98** | 扩散 / 多数 ComfyUI 视频流程主路径 |
| `per_block_mean=True`（非因果） | **~0.98** | 随机 QKV 冒烟仍接近 SDPA |
| 因果（`is_causal=True`） | **~0.7x** | mask **有效**；与 SDPA 贴合更粗 |
| 大 shape（如 L≥1024） | — | SA3 相对 SDPA 常 **2–7×** 更快；极短序列可能更慢（启动/量化开销） |

### 因果偏弱是 Windows 适配问题吗？

**不是。主要是 sageattn3 / FP4 算法与精度权衡，而不是 MSVC 独有的回归。**

依据：

1. **非因果 SA3 ≈ SDPA（~0.98）**（同一 Windows wheel）→ 量化、TMA、epilogue 以及 MSVC 指针传参 launch 整体健康。若 host launch 坏了，不会只“放过”非因果。
2. **因果 SA3 与非因果 SA3 差很大** → `is_causal` 已生效（不是 flag 没传 / 编错分支）。
3. **给 SDPA 做同样的 Q/K 均值中心化也填不平因果 gap** → 残差主要来自 **FP4 Q/K/V + 大量 `-inf` mask 下的 online score 路径**，不是“单纯忘了中心化”。
4. Windows 相关改动（指针传参、`sage_neg_inf_f()` 位型、编译选项等）对因果/非因果 **共用**；仅 padding mask 的非对齐长度仍与 SDPA 接近。

**使用建议：** 质量优先场景用 **非因果** SA3。若 workflow **强依赖因果且需贴近 SDPA**，那些层请用 SDPA 或其他后端，不要期望 FP4 SA3 因果 ≈ 全精度 SDPA。

---

## 使用方法

```python
from sageattn3 import sageattn3_blackwell

# q, k, v: FP16 或 BF16，形状 (batch, heads, seq_len, head_dim)
# head_dim 仅支持 64 或 128（>=256 会回退 SDPA）
out = sageattn3_blackwell(
    q, k, v,
    is_causal=False,
    per_block_mean=False,  # True：按 128 token 做分组均值中心化
)
```

### ComfyUI（KJNodes）

1. 将 wheel 装进 ComfyUI 使用的 Python。
2. 重启 ComfyUI。
3. **PatchSageAttentionKJ** 节点可选：
   - `sageattn3` → `per_block_mean=False`
   - `sageattn3_per_block_mean` → `per_block_mean=True`（本 fork **不依赖 Triton**）

### 布局说明

API 使用 **HND**：`(B, H, L, D)`。  
KJNodes 会在需要时从 Comfy 的 NHD 做转置。

---

## Windows 修复说明

相对上游 SageAttention3 Blackwell，本仓库主要改动：

| 方面 | 修复内容 |
|------|----------|
| MSVC 内核启动 | 超对齐 kernel 参数改为 **设备端指针** 传递（`DeviceParamsPack`）；MSVC 不能按值传 `alignas(128)` / `CUTE_GRID_CONSTANT` |
| TMA 描述符 | mainloop / epilogue 的 TMA 字段使用 `alignas(TMA_*)`，避免 `prefetch_tma_descriptor` 错位 |
| 调度器 | 使用真实 `multiProcessorCount`，不再写死 `170` SM |
| 编译选项 | C++20、`/Zc:__cplusplus`、`/Zc:preprocessor`、`/bigobj`、`-Usmall`、`NOMINMAX`、release `-O3`/`NDEBUG`、架构专用 `sm_XXXa` |
| 干净编译 | 去掉 torch 的 half 禁用 `-D`（避免 MSVC `D9025`）；IEEE 位型 ±inf（避免 nvcc `#221`）；无符号 shuffle mask（避免 `#68`）；抑制已知第三方头告警 |
| 头文件 | `#undef small`（Windows `rpcndr.h` 会把 `small` 定义成 `char`）；MSVC 安全的函数名宏 |
| 预处理 | 分组均值改用 **纯 PyTorch**，去掉 Triton，便于 ComfyUI embed 环境使用 `per_block_mean` |
| 工具链 | `build_wheel.bat` 优先 **`uv pip`**，把 venv 的 `Scripts` 加入 PATH 以找到 `ninja`，并在 Python ABI 切换时自动清理旧 `build\` |

实现参考了社区与上游相关讨论/PR：[#323](https://github.com/thu-ml/SageAttention/pull/323)、[#355](https://github.com/thu-ml/SageAttention/pull/355)、[#370](https://github.com/thu-ml/SageAttention/pull/370) 等。

可选调试构建（带 lineinfo / 更详细 ptxas）：

```bat
set SAGEATTN3_DEBUG=1
build_wheel.bat --clean
```

---

## 限制（与上游一致 + 实测补充）

SageAttention3 在多数 **图像生成** 模型以及部分 **视频** 模型（如 CogVideoX-2B、HunyuanVideo、Mochi）上效果较好，但 **不保证所有模型无损**。

建议：

- 优先保证 **head_dim 为 64 或 128**。
- 质量对齐 SDPA 时优先 **非因果** SA3（冒烟 cos≈0.98）；**因果** 视为更粗的 FP4 近似（见 [相对 SDPA 的质量说明](#相对-sdpa-的质量说明)）。
- 画质异常时，可对部分层/时间步混用 SageAttention2++，其余用 SageAttention3。
- 调试时 **不要** 设置 `CUDA_LAUNCH_BLOCKING=1`，可能干扰 TMA。
- **包名区分：** `sageattn3`（本仓库 / SA3）≠ `sageattention`（SA2）。部分 ComfyUI 节点（如 KJNodes MiniMax H3 *Mem Eff*）需要 **`sageattention` 2.x**，不是本 wheel。

---

## 故障排查

| 现象 | 处理 |
|------|------|
| 找不到 `cl.exe` / `vcvars` | 安装 **VS 2026** Build Tools + C++ 工作负载后重新运行 `build_wheel.bat` |
| MSVC 版本不符合预期 | 优先使用与当前 CUDA 配套的 VS 2026；脚本会自动探测 2026 / 2022 |
| 找不到 `nvcc` | 安装 CUDA Toolkit 12.8+，或使用 `--cuda` |
| `Unsupported GPU` | 仅支持 Blackwell（`sm_100` / `sm_120` / `sm_121`） |
| 安装成功但 import 失败 | 用与运行环境 **同一** Python 重新编译（`cp313` 与 `cp314t` 不可混用） |
| uv 环境提示 `No module named pip` | 使用 **`uv pip install ...`**，不要用 `python -m pip` |
| 自定义修改后又出现 `misaligned address` | 保留 MSVC 指针传参路径与 `/Zc:__cplusplus` |
| 安装后 ComfyUI 仍报错 | 彻底重启；确认包安装在 embed Python 路径下 |
| 编译内存不足 / 编译器崩溃 | `build_wheel.bat --jobs 1` |
| 3.13 ↔ 3.14t 切换后异常 | `build_wheel.bat --clean`（或依赖脚本自动清理旧 ABI 的 `build\`） |
| MiniMax / “Mem Eff Sage” 报 sageattention 版本错误 | 该节点要的是包 **`sageattention`**（SA2），不是 **`sageattn3`**。本 wheel 请用 KJ **Patch Sage Attention** 的 `sageattn3` 模式，或另行安装 SA2 |
| 因果与 SDPA 差很多 | FP4 SA3 因果的预期表现，不是 Windows launch 回归——见 [相对 SDPA 的质量说明](#相对-sdpa-的质量说明) |

---

## 目录结构

```text
Sageattention3_Blackwell_Windows/
  build_wheel.bat           # 一键 Windows 编译（MSVC + uv pip + wheel）
  setup.py                  # CUDA 扩展与 MSVC / nvcc 标志
  sageattn3/                # Python API 与 CUDA 源码
  scripts/smoke_vs_sdpa.py  # 相对 SDPA 的质量 / 速度 / 因果诊断
  LICENSE                   # Apache-2.0（上游）
  README.md
  README_zh.md
```

首次构建会把 CUTLASS 克隆到 `csrc/cutlass/`（已写入 `.gitignore`）。本地 `build/`、`dist/`、`*.log` 已忽略。

---

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。  
上游版权归属 SageAttention team / THU-ML。

---

## 引用

若使用 SageAttention3，请引用原作者：

```bibtex
@article{zhang2025sageattention3,
  title={SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training},
  author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Xu, Xiaoming and Huang, Haofeng and Wang, Haoxu and Jiang, Kai and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2505.11594},
  year={2025}
}
```

更多引用见[上游 README](https://github.com/thu-ml/SageAttention)。

---

## 免责声明

这是 **非官方** 的 Windows 打包与可移植性补丁集合，**不是** THU-ML 官方发行版。  
Python 包名仍为 `sageattn3`，以便与 ComfyUI-KJNodes 等现有工具直接兼容。
