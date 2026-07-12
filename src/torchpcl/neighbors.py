"""Unified nearest-neighbor search for single and packed point clouds."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .cloud import PointCloud
from .search import BatchedNearestNeighborSearch, NearestNeighborSearch


@dataclass(frozen=True, eq=False)
class Neighbors:
    """Fixed-width neighbor search results.

    Indices refer to global rows in the packed reference cloud. Invalid slots
    use index ``-1``, have an explicit false validity mask, and currently hold
    infinite squared distance.
    """

    indices: torch.Tensor
    distances2: torch.Tensor
    valid: torch.Tensor


def _as_cloud(value: torch.Tensor | PointCloud, name: str) -> PointCloud:
    if isinstance(value, PointCloud):
        return value
    if isinstance(value, torch.Tensor):
        return PointCloud.from_points(value)
    raise TypeError(f"{name} must be a torch.Tensor or PointCloud")


class NeighborIndex:
    """Reusable exact neighbor index over a packed reference cloud.

    ``algorithm="auto"`` uses BVH for a single reference cloud and packed
    brute force for batches. The current BVH backend cannot represent multiple
    independent clouds, so explicit ``"bvh"`` is restricted to batch size one.
    """

    def __init__(
        self,
        reference: torch.Tensor | PointCloud,
        *,
        algorithm: str = "auto",
    ) -> None:
        self._reference = _as_cloud(reference, "reference")
        if algorithm not in {"auto", "bvh", "bruteforce"}:
            raise ValueError("algorithm must be 'auto', 'bvh', or 'bruteforce'")
        if algorithm == "bvh" and self._reference.batch_size != 1:
            raise ValueError("the bvh backend currently supports one reference cloud")
        self._algorithm = (
            "bvh"
            if algorithm == "auto" and self._reference.batch_size == 1
            else "bruteforce" if algorithm == "auto" else algorithm
        )
        self._backends: dict[float, object] = {}

    @property
    def reference(self) -> PointCloud:
        return self._reference

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def _backend(self, radius: float):
        backend = self._backends.get(radius)
        if backend is not None:
            return backend
        points = self._reference.points.detach()
        if self._algorithm == "bvh":
            backend = NearestNeighborSearch(points, radius, backend="bvh")
        else:
            backend = BatchedNearestNeighborSearch(
                points, self._reference.offsets, radius
            )
        self._backends[radius] = backend
        return backend

    def _validate_queries(self, queries: torch.Tensor | PointCloud) -> PointCloud:
        cloud = _as_cloud(queries, "queries")
        if cloud.batch_size != self._reference.batch_size:
            raise ValueError("queries and reference must have the same batch size")
        if cloud.device != self._reference.device:
            raise ValueError("queries and reference must be on the same device")
        if cloud.dtype != self._reference.dtype:
            raise ValueError("queries and reference must have the same dtype")
        ref_lengths = self._reference.lengths
        query_lengths = cloud.lengths
        invalid_empty = torch.any((ref_lengths == 0) & (query_lengths > 0))
        if bool(invalid_empty):
            raise ValueError("a reference batch with queries must be non-empty")
        return cloud

    def _search(
        self,
        queries: torch.Tensor | PointCloud,
        *,
        k: int,
        radius: float,
    ) -> Neighbors:
        if not isinstance(k, int) or not 1 <= k <= 64:
            raise ValueError("k must be in [1, 64]")
        if radius <= 0 or math.isnan(radius):
            raise ValueError("radius must be positive")
        query_cloud = self._validate_queries(queries)
        if query_cloud.points.shape[0] == 0:
            shape = (0, k)
            return Neighbors(
                indices=torch.empty(shape, dtype=torch.int64, device=query_cloud.device),
                distances2=torch.empty(shape, dtype=query_cloud.dtype, device=query_cloud.device),
                valid=torch.empty(shape, dtype=torch.bool, device=query_cloud.device),
            )

        backend = self._backend(float(radius))
        if self._algorithm == "bvh":
            indices, _ = backend.knn_query(query_cloud.points.detach(), k)
        else:
            indices, _ = backend.knn_query(
                query_cloud.points.detach(), query_cloud.offsets, k
            )
        backend_valid = indices >= 0
        gathered = self._reference.points[indices.clamp(min=0)]
        distances2 = (query_cloud.points[:, None] - gathered).square().sum(dim=-1)
        valid = backend_valid & (distances2 <= radius * radius)
        indices = indices.masked_fill(~valid, -1)
        distances2 = distances2.masked_fill(~valid, math.inf)
        return Neighbors(indices=indices, distances2=distances2, valid=valid)

    def knn(self, queries: torch.Tensor | PointCloud, k: int) -> Neighbors:
        """Return the ``k`` nearest neighbors for every query."""
        return self._search(queries, k=k, radius=math.inf)

    def radius(
        self,
        queries: torch.Tensor | PointCloud,
        radius: float,
        *,
        max_neighbors: int = 64,
    ) -> Neighbors:
        """Return up to ``max_neighbors`` within ``radius``."""
        return self._search(queries, k=max_neighbors, radius=float(radius))

    def hybrid(
        self,
        queries: torch.Tensor | PointCloud,
        radius: float,
        k: int,
    ) -> Neighbors:
        """Return up to ``k`` nearest neighbors within ``radius``."""
        return self._search(queries, k=k, radius=float(radius))


def knn(
    reference: torch.Tensor | PointCloud,
    queries: torch.Tensor | PointCloud,
    k: int,
    *,
    algorithm: str = "auto",
) -> Neighbors:
    return NeighborIndex(reference, algorithm=algorithm).knn(queries, k)


def radius_neighbors(
    reference: torch.Tensor | PointCloud,
    queries: torch.Tensor | PointCloud,
    radius: float,
    *,
    max_neighbors: int = 64,
    algorithm: str = "auto",
) -> Neighbors:
    return NeighborIndex(reference, algorithm=algorithm).radius(
        queries, radius, max_neighbors=max_neighbors
    )


def hybrid_neighbors(
    reference: torch.Tensor | PointCloud,
    queries: torch.Tensor | PointCloud,
    radius: float,
    k: int,
    *,
    algorithm: str = "auto",
) -> Neighbors:
    return NeighborIndex(reference, algorithm=algorithm).hybrid(queries, radius, k)


__all__ = [
    "NeighborIndex",
    "Neighbors",
    "hybrid_neighbors",
    "knn",
    "radius_neighbors",
]
