"""cuBQL-backed nearest-neighbor search (CPU and CUDA).

torchpcl's spatial queries run on a cuBQL BVH (third_party/cuBQL). Both
backends use the same traversal code: a CUDA extension on GPU and a
plain C++ extension (parallelized over the intra-op thread pool) on
CPU. The extensions are compiled at install time by setup.py; the CUDA
one is absent when torchpcl was installed without a CUDA toolkit.
"""

import torch

from . import _cubql_cpu

try:
    from . import _cubql_cuda
except ImportError:  # installed without a CUDA toolkit
    _cubql_cuda = None


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
            if _cubql_cuda is None:
                raise RuntimeError(
                    "torchpcl was installed without the CUDA extension; "
                    "reinstall with a CUDA toolkit available (see README)"
                )
            ext = _cubql_cuda
        elif points.device.type == "cpu":
            ext = _cubql_cpu
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
