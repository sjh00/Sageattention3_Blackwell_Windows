# SageAttention3 Blackwell Windows 版

面向 **Windows** 的 **SageAttention3**（NVIDIA **Blackwell** GPU：RTX 50 系 / `sm_120`，以及 `sm_100` / `sm_121`）构建工程。

本仓库在官方 SageAttention3 Blackwell 源码基础上，加入 **MSVC / Windows 运行时修复**，使生成的 wheel **不仅能编译，还能稳定运行**（避免常见的 `CUDA error: misaligned address` 崩溃）。

> 上游项目：[thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)  
> 论文：[SageAttention3: Microscaling FP4 Attention](https://arxiv.org/abs/2505.11594)  
> English docs: [README.md](README.md)

---

## 为什么需要这个仓库

官方 SageAttention3 以 Linux 为主。在 Windows + MSVC 上常见问题：

1. **编译失败**（Windows 头文件里的 `small` 宏、`std` 歧义、C++ 标准开关等）。
2. **wheel 能装却一用就崩**：
   - `CUDA error: misaligned address`
   - 栈落在 `sageattn3_blackwell` / `fp4attn_cuda.fwd`
3. **Triton 在 ComfyUI 便携 Python 上不可用**（缺少 `Python.h`），导致 `per_block_mean=True` 失败。

本仓库提供一套已验证的 Windows 补丁（见 [Windows 修复说明](#windows-修复说明)）。参考成功环境：

| 项目 | 示例 |
|------|------|
| 系统 | Windows 11 |
| 显卡 | RTX 50 系（compute capability 12.0） |
| Python | 3.13 |
| PyTorch | 2.13 + CUDA 13.2 |
| CUDA Toolkit | 13.2 |
| MSVC | VS 2022 Build Tools 14.44 |
| 应用 | ComfyUI + KJNodes `PatchSageAttention` |

---

## 环境要求

### 硬件

- 仅支持 NVIDIA **Blackwell**：
  - `sm_100`
  - `sm_120`（GeForce RTX 50 系）
  - `sm_121`

### 软件

- **Python** 3.10+（推荐 3.12 / 3.13；需与 ComfyUI embed 或目标环境一致）
- **带 CUDA 的 PyTorch**（建议 torch >= 2.8，CUDA 12.8+）
- **CUDA Toolkit >= 12.8**（50 系建议 13.x），需有 `nvcc`，或通过 `CUDA_HOME` / `--cuda` 指定
- **Visual Studio 2022** Build Tools，勾选 **使用 C++ 的桌面开发**
- **Git**（首次构建会克隆 [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)）
- **ninja / packaging / wheel / build**（构建脚本会自动 `pip install`）

---

## 快速开始：编译 wheel

在本目录打开 `cmd`（脚本会自行加载 MSVC 环境）：

```bat
build_wheel.bat
```

常用参数：

```bat
rem 清理旧产物再编
build_wheel.bat --clean

rem 指定 ComfyUI 便携 Python
build_wheel.bat --python "D:\ComfyUI\python_embeded\python.exe"

rem 指定 CUDA Toolkit 路径
build_wheel.bat --cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

rem 并行编译任务数（默认 2，内存不够可改 1）
build_wheel.bat --jobs 1
```

成功后 wheel 在 `dist\`：

```text
dist\sageattn3-1.0.0-cpXXX-cpXXX-win_amd64.whl
```

完整编译日志：`build.log`。

### 手动编译（进阶）

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_HOME%\bin;%PATH%
python -m pip install -U pip setuptools wheel ninja packaging build
python -m build --wheel --no-isolation
```

---

## 安装 wheel

**构建用的 Python 与运行环境必须一致**（例如都是 `cp313`）：

```bat
python -m pip install --force-reinstall --no-deps dist\sageattn3-1.0.0-cp313-cp313-win_amd64.whl
```

ComfyUI 便携包示例：

```bat
D:\ComfyUI\python_embeded\python.exe -m pip install --force-reinstall --no-deps dist\sageattn3-1.0.0-cp313-cp313-win_amd64.whl
```

安装后请 **完全重启 ComfyUI**。

### 冒烟测试

```bat
python -c "import torch; from sageattn3 import sageattn3_blackwell; q=torch.randn(1,8,128,128,device='cuda',dtype=torch.bfloat16); print(sageattn3_blackwell(q,q,q,per_block_mean=False).shape)"
```

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
| 编译选项 | C++20、`/Zc:__cplusplus`、`/Zc:preprocessor`、`/bigobj`、`-Usmall`、`USE_CUDA` |
| 头文件 | `#undef small`（Windows `rpcndr.h` 会把 `small` 定义成 `char`） |
| 预处理 | 分组均值改用 **纯 PyTorch**，去掉 Triton，便于 ComfyUI embed 环境使用 `per_block_mean` |

实现参考了社区与上游相关讨论/PR：[#323](https://github.com/thu-ml/SageAttention/pull/323)、[#355](https://github.com/thu-ml/SageAttention/pull/355)、[#370](https://github.com/thu-ml/SageAttention/pull/370) 等。

---

## 限制（与上游一致）

SageAttention3 在多数 **图像生成** 模型以及部分 **视频** 模型（如 CogVideoX-2B、HunyuanVideo、Mochi）上效果较好，但 **不保证所有模型无损**。

建议：

- 优先保证 **head_dim 为 64 或 128**。
- 画质异常时，可对部分层/时间步混用 SageAttention2++，其余用 SageAttention3。
- 调试时 **不要** 设置 `CUDA_LAUNCH_BLOCKING=1`，可能干扰 TMA。

---

## 故障排查

| 现象 | 处理 |
|------|------|
| 找不到 `cl.exe` / `vcvars` | 安装 VS 2022 C++ 工作负载后重新运行 `build_wheel.bat` |
| 找不到 `nvcc` | 安装 CUDA Toolkit 12.8+，或使用 `--cuda` |
| `Unsupported GPU` | 仅支持 Blackwell（`sm_100` / `sm_120` / `sm_121`） |
| 安装成功但 import 失败 | 用与运行环境 **同一** Python 重新编译（cp312 / cp313 不可混用） |
| 自定义修改后又出现 `misaligned address` | 保留 MSVC 指针传参路径与 `/Zc:__cplusplus` |
| 安装后 ComfyUI 仍报错 | 彻底重启；确认 `pip show sageattn3` 路径在 `python_embeded` 下 |
| 编译内存不足 / 编译器崩溃 | `build_wheel.bat --jobs 1` |

---

## 目录结构

```text
sageattention3_blackwell_Windows/
  build_wheel.bat      # 一键 Windows 编译
  setup.py             # CUDA 扩展与 MSVC 标志
  sageattn3/           # Python API 与 CUDA 源码
  LICENSE              # Apache-2.0（上游）
  README.md
  README_zh.md
```

首次构建会把 CUTLASS 克隆到 `csrc/cutlass/`（已写入 `.gitignore`）。

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
