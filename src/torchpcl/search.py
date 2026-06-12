"""cuBQL-backed nearest-neighbor search (CPU and CUDA).

torchpcl's spatial queries run on a cuBQL BVH (third_party/cuBQL). Both
backends use the same traversal code: a CUDA extension on GPU and a
plain C++ extension (parallelized over the intra-op thread pool) on
CPU. Extensions are JIT-compiled on first use and cached by torch;
torchpcl therefore requires a dev checkout with the cuBQL headers, a
C++ compiler, and -- for the CUDA path -- the nvcc toolchain installed
via the project dependencies.
"""

import functools
import os
from pathlib import Path

import torch

_EXT_NAME = "torchpcl_cubql"
_CPU_EXT_NAME = "torchpcl_cubql_cpu"


def _find_cubql_root() -> Path:
    override = os.environ.get("TORCHPCL_CUBQL_DIR")
    candidates = (
        [Path(override)] if override
        else [Path(__file__).resolve().parents[2] / "third_party" / "cuBQL"]
    )
    for root in candidates:
        if (root / "cuBQL" / "bvh.h").is_file():
            return root
    raise RuntimeError(
        "cuBQL headers not found. torchpcl requires a dev checkout: "
        "clone/init third_party/cuBQL in the torchpcl repository, or point "
        "TORCHPCL_CUBQL_DIR at a cuBQL checkout."
    )


def _find_cuda_home() -> Path:
    """Locate a CUDA toolkit matching torch's CUDA major version.

    Prefers the pip-installed toolkit (nvidia/cu<major> in site-packages,
    installed via `uv sync`); falls back to CUDA_HOME/CUDA_PATH if it
    points at a toolkit with nvcc.
    """
    major = torch.version.cuda.split(".")[0]
    try:
        import nvidia
        # namespace package: locate via __path__ (__file__ is None)
        for nvidia_dir in nvidia.__path__:
            pip_home = Path(nvidia_dir) / f"cu{major}"
            if (pip_home / "bin" / "nvcc").is_file():
                return pip_home
    except ImportError:
        pass

    env_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_home and (Path(env_home) / "bin" / "nvcc").is_file():
        return Path(env_home)

    raise RuntimeError(
        "No CUDA toolkit with nvcc found. Install the pip toolchain with "
        "`uv sync`, or set CUDA_HOME to a toolkit matching torch's CUDA "
        f"version ({torch.version.cuda})."
    )


def _cudart_shim_dir(cuda_home: Path) -> Path:
    """Pip CUDA wheels ship only libcudart.so.<major>; the linker needs an
    unversioned libcudart.so. Maintain a symlink in a cache dir we own."""
    major = torch.version.cuda.split(".")[0]
    target = cuda_home / "lib" / f"libcudart.so.{major}"
    shim_dir = Path(os.path.expanduser("~/.cache/torchpcl/lib"))
    shim_dir.mkdir(parents=True, exist_ok=True)
    link = shim_dir / "libcudart.so"
    if link.resolve() != target.resolve():
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    return shim_dir


@functools.cache
def _load_cuda_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available for the CUDA search extension")
    if torch.version.cuda is None:
        raise RuntimeError("the CUDA search extension requires a CUDA build of torch")

    cubql_root = _find_cubql_root()
    cuda_home = _find_cuda_home()

    # cpp_extension snapshots CUDA_HOME at import time -- set it first.
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    # Build only for the local device. A user-wide TORCH_CUDA_ARCH_LIST may
    # name archs the pinned nvcc no longer supports (CUDA 13 dropped
    # compute_61), so it is deliberately overridden for this extension;
    # use TORCHPCL_CUDA_ARCH_LIST to compile for other archs.
    capability = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get(
        "TORCHPCL_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}"
    )

    from torch.utils import cpp_extension

    if cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = str(cuda_home)
    if not cpp_extension.is_ninja_available():
        raise RuntimeError(
            "ninja is required to build the torchpcl extension; "
            "install it with `uv sync`"
        )

    sources = [str(Path(__file__).resolve().parent / "csrc" / "cubql_search.cu")]
    include_paths = [str(cubql_root), str(cuda_home / "include")]
    cccl = cuda_home / "include" / "cccl"
    if cccl.is_dir():
        include_paths.append(str(cccl))

    return cpp_extension.load(
        name=_EXT_NAME,
        sources=sources,
        extra_include_paths=include_paths,
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{_cudart_shim_dir(cuda_home)}"],
    )


@functools.cache
def _load_cpu_extension():
    cubql_root = _find_cubql_root()

    from torch.utils import cpp_extension

    if not cpp_extension.is_ninja_available():
        raise RuntimeError(
            "ninja is required to build the torchpcl extension; "
            "install it with `uv sync`"
        )

    sources = [str(Path(__file__).resolve().parent / "csrc" / "cubql_search_cpu.cpp")]
    # -fopenmp is required for at::parallel_for to actually parallelize:
    # ATen's OpenMP backend is header-inline, so without the flag the
    # pragmas vanish and queries run single-threaded.
    return cpp_extension.load(
        name=_CPU_EXT_NAME,
        sources=sources,
        extra_include_paths=[str(cubql_root)],
        extra_cflags=["-O3", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
    )


class NearestNeighborSearch:
    """1-NN search within a fixed radius over a static point set.

    Matches Open3D's hybrid search semantics: for each query point, the
    nearest point within ``radius``, or -1 if none exists. The radius may
    be ``math.inf`` for unbounded search. The BVH lives on the device of
    ``points`` (CPU or CUDA); queries must be on the same device.
    """

    def __init__(self, points: torch.Tensor, radius: float):
        self._radius = float(radius)
        self._points_f32 = points.to(torch.float32).contiguous()
        if points.device.type == "cuda":
            ext = _load_cuda_extension()
        elif points.device.type == "cpu":
            ext = _load_cpu_extension()
        else:
            raise RuntimeError(
                f"unsupported device '{points.device}' for torchpcl spatial search"
            )
        self._bvh = ext.PointBVH(self._points_f32)

    def query(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (indices, dist2) per query point; index -1 = no neighbor
        within the radius, dist2 only meaningful where index >= 0."""
        queries_f32 = queries.to(torch.float32).contiguous()
        indices, dist2 = self._bvh.query(queries_f32, self._radius)
        return indices.to(torch.int64), dist2

    def knn_query(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return up to k nearest neighbors within the radius per query.

        indices: (M, k) int64 sorted by distance, -1 padded; k <= 64.
        dist2: (M, k) float32 squared distances, meaningful where
            the index is >= 0.
        """
        queries_f32 = queries.to(torch.float32).contiguous()
        indices, dist2 = self._bvh.knn(queries_f32, k, self._radius)
        return indices.to(torch.int64), dist2


__all__ = ["NearestNeighborSearch"]
