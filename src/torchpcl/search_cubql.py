"""cuBQL-backed nearest-neighbor search (CUDA-only, experimental).

Drop-in alternative to :class:`torchpcl.search.NearestNeighborSearch`,
using a GPU-built BVH (third_party/cuBQL) instead of a warp hash grid.
The CUDA extension is JIT-compiled on first use and cached by torch;
this is a dev-checkout feature, not part of a wheel install.
"""

import functools
import os
from pathlib import Path

import torch

_EXT_NAME = "torchpcl_cubql"


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
        "cuBQL headers not found. The cubql backend is a dev-checkout feature: "
        "clone/init third_party/cuBQL in the torchpcl repository, or point "
        "TORCHPCL_CUBQL_DIR at a cuBQL checkout."
    )


def _find_cuda_home() -> Path:
    """Locate a CUDA toolkit matching torch's CUDA major version.

    Prefers the pip-installed toolkit (nvidia/cu<major> in site-packages,
    installed via `uv sync --group cubql`); falls back to a system toolkit
    only if torch's own detection found one with a matching major version.
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
        "No CUDA toolkit with nvcc found for the cubql backend. Install the "
        "pip toolchain with `uv sync --group cubql`, or set CUDA_HOME to a "
        f"toolkit matching torch's CUDA version ({torch.version.cuda})."
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
def _load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("the cubql backend requires a CUDA device")
    if torch.version.cuda is None:
        raise RuntimeError("the cubql backend requires a CUDA build of torch")

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
            "ninja is required to build the cubql backend; "
            "install it with `uv sync --group cubql`"
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


class CuBQLNearestNeighborSearch:
    """1-NN search within a fixed radius over a static point set (CUDA).

    Same interface and semantics as the warp-backed
    :class:`torchpcl.search.NearestNeighborSearch`.
    """

    def __init__(self, points: torch.Tensor, radius: float):
        if points.device.type != "cuda":
            raise RuntimeError(
                f"the cubql backend is CUDA-only; got points on '{points.device}'"
            )
        self._radius = float(radius)
        self._points_f32 = points.to(torch.float32).contiguous()
        self._bvh = _load_extension().PointBVH(self._points_f32)

    def query(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (indices, dist2) per query point; index -1 = no neighbor
        within the radius, dist2 only meaningful where index >= 0."""
        queries_f32 = queries.to(torch.float32).contiguous()
        indices, dist2 = self._bvh.query(queries_f32, self._radius)
        return indices.to(torch.int64), dist2

    def knn_query(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return up to k nearest neighbors within the radius per query.

        Same interface as the warp backend's knn_query, but the radius may
        be ``math.inf`` for unbounded (true) k-NN. k is capped at 64.
        """
        queries_f32 = queries.to(torch.float32).contiguous()
        indices, dist2 = self._bvh.knn(queries_f32, k, self._radius)
        return indices.to(torch.int64), dist2
