"""Build the private torchpcl search extension at installation time."""

import os
import sys
import warnings
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import (  # noqa: E402
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "src" / "torchpcl" / "csrc"


def find_cubql_root() -> Path:
    override = os.environ.get("TORCHPCL_CUBQL_DIR")
    root = Path(override) if override else ROOT / "third_party" / "cuBQL"
    if not (root / "cuBQL" / "bvh.h").is_file():
        raise RuntimeError(
            "cuBQL headers not found; initialize third_party/cuBQL or set "
            "TORCHPCL_CUBQL_DIR"
        )
    return root


def cuda_enabled() -> bool:
    mode = os.environ.get("TORCHPCL_WITH_CUDA", "auto").lower()
    if mode not in {"auto", "0", "1"}:
        raise RuntimeError("TORCHPCL_WITH_CUDA must be 'auto', '0', or '1'")
    toolkit = CUDA_HOME is not None and (Path(CUDA_HOME) / "bin" / "nvcc").is_file()
    if mode == "1" and not toolkit:
        raise RuntimeError(
            "TORCHPCL_WITH_CUDA=1 but a CUDA toolkit with nvcc was not found"
        )
    enabled = toolkit if mode == "auto" else mode == "1"
    variant = "CUDA" if enabled else "CPU-only"
    print(f"torchpcl: building {variant} torchpcl._C extension")
    if mode == "auto" and not toolkit and torch.version.cuda is not None:
        warnings.warn(
            "installed PyTorch supports CUDA, but no CUDA toolkit was found; "
            "building torchpcl CPU-only. Set TORCHPCL_WITH_CUDA=1 to require CUDA.",
            stacklevel=2,
        )
    return enabled


def parallel_flags() -> tuple[list[str], list[str]]:
    # ATen's OpenMP backend is header-inline. Without -fopenmp,
    # at::parallel_for silently becomes single-threaded on Linux.
    if sys.platform.startswith("linux"):
        return ["-O3", "-fopenmp"], ["-fopenmp"]
    return ["-O3"], []


cubql_root = find_cubql_root()
cxx_flags, link_flags = parallel_flags()
include_dirs = [str(CSRC), str(cubql_root)]
cpu_sources = [
    "src/torchpcl/csrc/bindings.cpp",
    "src/torchpcl/csrc/cpu/bvh.cpp",
    "src/torchpcl/csrc/cpu/packed_knn.cpp",
]

if cuda_enabled():
    extension = CUDAExtension(
        "torchpcl._C",
        cpu_sources
        + [
            "src/torchpcl/csrc/cuda/bvh.cu",
            "src/torchpcl/csrc/cuda/packed_knn.cu",
        ],
        include_dirs=include_dirs,
        define_macros=[("TORCHPCL_WITH_CUDA", "1")],
        extra_compile_args={"cxx": cxx_flags, "nvcc": ["-O3"]},
        extra_link_args=link_flags,
    )
else:
    extension = CppExtension(
        "torchpcl._C",
        cpu_sources,
        include_dirs=include_dirs,
        extra_compile_args=cxx_flags,
        extra_link_args=link_flags,
    )

setup(
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
