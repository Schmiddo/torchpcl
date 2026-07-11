"""Build the cuBQL search extensions (src/torchpcl/csrc) at install time.

The build compiles against the torch installed in the environment (and,
for the CUDA extension, an nvcc matching torch's CUDA version), so
install with build isolation disabled. Without a CUDA toolkit the CUDA
extension is skipped and torchpcl runs CPU-only.
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import (  # noqa: E402
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

ROOT = Path(__file__).resolve().parent
CPU_ONLY = os.environ.get("TORCHPCL_CPU_ONLY")


def find_cubql_root() -> Path:
    override = os.environ.get("TORCHPCL_CUBQL_DIR")
    root = Path(override) if override else ROOT / "third_party" / "cuBQL"
    if not (root / "cuBQL" / "bvh.h").is_file():
        raise RuntimeError(
            "cuBQL headers not found; clone/init third_party/cuBQL or point "
            "TORCHPCL_CUBQL_DIR at a cuBQL checkout"
        )
    return root

cubql_root = find_cubql_root()

ext_modules = [
    # -fopenmp is required for at::parallel_for to actually parallelize:
    # ATen's OpenMP backend is header-inline, so without the flag the
    # pragmas vanish and queries run single-threaded.
    CppExtension(
        "torchpcl._cubql_cpu",
        ["src/torchpcl/csrc/cubql_search_cpu.cpp"],
        include_dirs=[str(cubql_root)],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

if (cuda_home := os.environ.get("CUDA_HOME")) is not None:
    include_dirs = [str(cubql_root), str(Path(cuda_home) / "include")]
    ext_modules.append(
        CUDAExtension(
            "torchpcl._cubql_cuda",
            ["src/torchpcl/csrc/cubql_search.cu"],
            include_dirs=include_dirs,
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    )
else:
    if CPU_ONLY:
        print("torchpcl: no CUDA toolkit with nvcc found; building without the CUDA extension")
    else:
        raise RuntimeError(
            "Could not find CUDA_HOME. "
            "Either provide $CUDA_HOME or install with TORCHPCL_CPU_ONLY=1"
        )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}}
)
