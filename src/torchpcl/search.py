"""Exact nearest-neighbor search on CPU and CUDA.

The default backend uses a cuBQL BVH. A standalone tiled brute-force backend
is available for k-NN and packed ragged batches. Extensions are compiled at
install time by setup.py; CUDA modules are absent when torchpcl was installed
without a CUDA toolkit.
"""

import torch

from . import _cubql_cpu
from . import _bruteforce_cpu

try:
    from . import _cubql_cuda
except ImportError:  # installed without a CUDA toolkit
    _cubql_cuda = None

try:
    from . import _bruteforce_cuda
except ImportError:  # installed without a CUDA toolkit
    _bruteforce_cuda = None


def _extension(device: torch.device, backend: str):
    if backend == "auto":
        backend = "bvh"
    if backend not in {"bvh", "bruteforce"}:
        raise ValueError("backend must be 'auto', 'bvh', or 'bruteforce'")
    if device.type == "cpu":
        return (_cubql_cpu if backend == "bvh" else _bruteforce_cpu), backend
    if device.type == "cuda":
        ext = _cubql_cuda if backend == "bvh" else _bruteforce_cuda
        if ext is None:
            raise RuntimeError(
                f"torchpcl was installed without the CUDA {backend} extension; "
                "reinstall with a CUDA toolkit available (see README)"
            )
        return ext, backend
    raise RuntimeError(f"unsupported device '{device}' for torchpcl spatial search")


def _single_offsets(size: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([0, size], dtype=torch.int64, device=device)


class NearestNeighborSearch:
    """1-NN search within a fixed radius over a static point set.

    Matches Open3D's hybrid search semantics: for each query point, the
    nearest point within ``radius``, or -1 if none exists. The radius may
    be ``math.inf`` for unbounded search. Indexed data lives on the device of
    ``points`` (CPU or CUDA); queries must be on the same device.
    """

    def __init__(
        self, points: torch.Tensor, radius: float, *, backend: str = "bvh"
    ):
        self._radius = float(radius)
        self._points_f32 = points.to(torch.float32).contiguous()
        self._ext, self._backend = _extension(points.device, backend)
        if self._backend == "bvh":
            self._index = self._ext.PointBVH(self._points_f32)
        else:
            if self._points_f32.shape[0] == 0:
                raise ValueError("points must be non-empty")
            self._point_offsets = _single_offsets(
                self._points_f32.shape[0], self._points_f32.device
            )

    def query(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (indices, dist2) per query point; index -1 = no neighbor
        within the radius, dist2 only meaningful where index >= 0."""
        queries_f32 = queries.to(torch.float32).contiguous()
        if self._backend == "bvh":
            indices, dist2 = self._index.query(queries_f32, self._radius)
        else:
            query_offsets = _single_offsets(queries_f32.shape[0], queries_f32.device)
            indices, dist2 = self._ext.knn(
                self._points_f32,
                self._point_offsets,
                queries_f32,
                query_offsets,
                1,
                self._radius,
            )
            indices, dist2 = indices[:, 0], dist2[:, 0]
        return indices.to(torch.int64), dist2

    def knn_query(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return up to k nearest neighbors within the radius per query.

        indices: (M, k) int64 sorted by distance, -1 padded; k <= 64.
        dist2: (M, k) float32 squared distances, meaningful where
            the index is >= 0.
        """
        queries_f32 = queries.to(torch.float32).contiguous()
        if self._backend == "bvh":
            indices, dist2 = self._index.knn(queries_f32, k, self._radius)
        else:
            query_offsets = _single_offsets(queries_f32.shape[0], queries_f32.device)
            indices, dist2 = self._ext.knn(
                self._points_f32,
                self._point_offsets,
                queries_f32,
                query_offsets,
                k,
                self._radius,
            )
        return indices.to(torch.int64), dist2


class BatchedNearestNeighborSearch:
    """Exact k-NN over packed ragged batches using tiled brute force.

    Offsets have shape ``(B + 1,)`` and delimit contiguous batches. Returned
    indices are global rows in the packed ``points`` tensor.
    """

    def __init__(
        self, points: torch.Tensor, point_offsets: torch.Tensor, radius: float
    ):
        self._radius = float(radius)
        self._points_f32 = points.to(torch.float32).contiguous()
        self._point_offsets = point_offsets.to(
            device=points.device, dtype=torch.int64
        ).contiguous()
        self._ext, _ = _extension(points.device, "bruteforce")

    def knn_query(
        self, queries: torch.Tensor, query_offsets: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries_f32 = queries.to(torch.float32).contiguous()
        offsets = query_offsets.to(
            device=queries.device, dtype=torch.int64
        ).contiguous()
        indices, dist2 = self._ext.knn(
            self._points_f32,
            self._point_offsets,
            queries_f32,
            offsets,
            k,
            self._radius,
        )
        return indices.to(torch.int64), dist2

    def query(
        self, queries: torch.Tensor, query_offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices, dist2 = self.knn_query(queries, query_offsets, 1)
        return indices[:, 0], dist2[:, 0]


__all__ = ["BatchedNearestNeighborSearch", "NearestNeighborSearch"]
