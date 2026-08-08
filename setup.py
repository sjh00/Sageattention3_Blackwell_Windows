import warnings
import os
import sys
from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages
import subprocess

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - older setuptools
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "sageattn3"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FAHOPPER_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FAHOPPER_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FAHOPPER_FORCE_CXX11_ABI", "FALSE") == "TRUE"
# SAGEATTN3_DEBUG=1 keeps lineinfo + verbose ptxas (slower kernels)
DEBUG_BUILD = os.getenv("SAGEATTN3_DEBUG", "FALSE").upper() in ("1", "TRUE", "YES")


def get_cuda_bare_metal_version(cuda_dir):
    # Use pathlib so Windows paths with spaces work; require nvcc.exe on NT.
    nvcc = Path(cuda_dir) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    if not nvcc.is_file():
        nvcc = Path(cuda_dir) / "bin" / "nvcc"
    raw_output = subprocess.check_output([str(nvcc), "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    # Respect MAX_JOBS for nvcc internal parallelism when set.
    try:
        threads = max(1, int(os.getenv("MAX_JOBS", "4")))
    except ValueError:
        threads = 4
    threads = min(threads, 8)
    return nvcc_extra_args + ["--threads", str(threads)]


def strip_torch_half_disable_flags():
    """Torch injects -D__CUDA_NO_HALF_* which we used to counteract with -U...

    That produces MSVC command-line warning D9025 (redefine /D with /U). Removing the
    -D flags instead keeps half/bfloat ops available for CUTLASS with a clean cmdline.
    """
    remove = {
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    }
    try:
        cpp_extension.COMMON_NVCC_FLAGS = [
            f for f in cpp_extension.COMMON_NVCC_FLAGS if f not in remove
        ]
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not strip torch COMMON_NVCC_FLAGS: {exc}")


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("sageattn3")
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set and nvcc was not found. "
            "Install CUDA Toolkit >= 12.8 and set CUDA_HOME, or pass --cuda to build_wheel.bat."
        )

    strip_torch_half_disable_flags()

    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required at build time to select sm_100/sm_120/sm_121")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    # Use arch-specific "a" variants for full Blackwell feature set / best perf.
    if (cc_major, cc_minor) == (10, 0):  # sm_100
        cc_flag += ["-gencode", "arch=compute_100a,code=sm_100a"]
    elif (cc_major, cc_minor) == (12, 0):  # sm_120
        cc_flag += ["-gencode", "arch=compute_120a,code=sm_120a"]
    elif (cc_major, cc_minor) == (12, 1):  # sm_121
        cc_flag += ["-gencode", "arch=compute_121a,code=sm_121a"]
    else:
        raise RuntimeError(
            f"Unsupported GPU compute capability {cc_major}.{cc_minor}; "
            "need sm_100 / sm_120 / sm_121 (Blackwell)"
        )
    print(f"[sageattn3] target GPU sm_{cc_major}{cc_minor}  CUDA Toolkit {bare_metal_version}")

    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    repo_dir = Path(this_dir)
    cutlass_dir = repo_dir / "csrc" / "cutlass"
    (repo_dir / "csrc").mkdir(parents=True, exist_ok=True)
    if not cutlass_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", str(cutlass_dir)],
            check=True,
        )

    nvcc_flags = [
        "-O3",
        "-std=c++20",
        # Half/bf16 ops must stay enabled for CUTLASS FP4 paths (torch -D flags stripped above).
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",  # Critical: without this, CUTLASS asserts tank performance
        "-DQBLKSIZE=128",
        "-DKBLKSIZE=128",
        "-DCTA256",
        "-DDQINRMEM",
        # Third-party headers only (torch / cutlass) — keep our sources warning-clean.
        # 3189: torch "module" token in cudafe
        # 2908: cutlass deprecated implicit this capture
        "-diag-suppress=3189,2908",
        "-Xcudafe",
        "--diag_suppress=3189",
        "-Xcudafe",
        "--diag_suppress=2908",
    ]
    if DEBUG_BUILD:
        nvcc_flags += [
            "-lineinfo",
            "--ptxas-options=--verbose,--warn-on-local-memory-usage",
        ]
    else:
        # Release: no lineinfo (better perf). ptxas local-mem warnings are informational
        # for these large Blackwell kernels and not actionable — keep logs clean.
        pass

    library_dirs = []
    # /Zc:__cplusplus is critical on MSVC: without it __cplusplus stays 199711L and
    # CUTLASS/CUTE alignment macros (CUTE_GRID_CONSTANT etc.) are disabled, which
    # produces kernels that crash at runtime with CUDA misaligned address.
    if os.name == "nt":
        cxx_flags = [
            "/std:c++20",
            "/Zc:__cplusplus",
            "/Zc:preprocessor",
            "/DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING",
            "/bigobj",
            "/MD",
            "/permissive-",
            "/O2",
            "/DNDEBUG",
            # Host-side noise from long TUs / third-party headers
            "/wd4819",  # code page
            "/wd4624",  # destructor was implicitly defined as deleted
            "/wd4068",  # unknown pragma
            "/wd4251",  # dll-interface
            "/wd5285",  # cutlass uses non-specialized is_reference traits under MSVC
        ]
        nvcc_flags += [
            "-D_WIN32=1",
            "-DUSE_CUDA=1",
            # Windows headers define min/max/small macros; neutralize for CUTLASS.
            "-Usmall",
            "-DNOMINMAX",
            "-DWIN32_LEAN_AND_MEAN",
        ]
        nvcc_flags += [f"-Xcompiler={flag}" for flag in cxx_flags]
        cuda_lib = Path(CUDA_HOME) / "lib" / "x64"
        if cuda_lib.is_dir():
            library_dirs.append(str(cuda_lib))
    else:
        cxx_flags = ["-O3", "-std=c++20", "-DNDEBUG"]

    include_dirs = [
        repo_dir / "sageattn3",
        cutlass_dir / "include",
        cutlass_dir / "tools" / "util" / "include",
    ]

    def make_ext_kwargs():
        kwargs = dict(
            extra_compile_args={
                "cxx": list(cxx_flags),
                "nvcc": append_nvcc_threads(list(nvcc_flags) + ["-DEXECMODE=0"] + list(cc_flag)),
            },
            include_dirs=list(include_dirs),
            library_dirs=list(library_dirs),
            # cuTensorMapEncodeTiled lives in cuda.lib (driver API)
            libraries=["cuda"],
        )
        if os.name == "nt":
            # Avoid linker chatter when LTCG is requested but no /GL objects need it.
            kwargs["extra_link_args"] = ["/LTCG:OFF"]
        return kwargs

    if hasattr(sys, "_is_gil_enabled"):
        print(f"[sageattn3] free-threaded build detected: {sys.version.splitlines()[0]}")

    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=["sageattn3/blackwell/api.cu"],
            **make_ext_kwargs(),
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="fp4quant_cuda",
            sources=["sageattn3/quantization/fp4_quantization_4d.cu"],
            **make_ext_kwargs(),
        )
    )


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        super().run()


# Prefer ninja for stable parallel compiles on Windows.
_build_ext = BuildExtension.with_options(use_ninja=True) if ext_modules else None

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="SageAttention3 Blackwell FP4 attention (Windows-ready build)",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    ext_modules=ext_modules,
    cmdclass=(
        {"bdist_wheel": CachedWheelsCommand, "build_ext": _build_ext}
        if ext_modules
        else {"bdist_wheel": CachedWheelsCommand}
    ),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
