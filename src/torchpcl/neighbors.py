"""Unified nearest-neighbor search for single and packed point clouds."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._backend import BvhBackend, PackedBackend
from .cloud import PointCloud, PointCloudLike, as_point_cloud
from .validation import check_cloud_pair


@dataclass(frozen=True, eq=False)
class Neighbors:
    """Fixed-width neighbor search results.

    Indices refer to global rows in the packed reference cloud. Invalid slots
    use index ``-1``, have an explicit false validity mask, and currently hold
    infinite squared distance. Neighbor order and distance-tie resolution are
    intentionally unspecified.
    """

    indices: torch.Tensor
    distances2: torch.Tensor
    valid: torch.Tensor


class NeighborIndex:
    """Reusable exact neighbor index over a packed reference cloud.

    ``algorithm="auto"`` uses BVH for a single reference cloud and packed
    brute force for batches. The current BVH backend cannot represent multiple
    independent clouds, so explicit ``"bvh"`` is restricted to batch size one.
    """

    def __init__(
        self,
        reference: PointCloudLike,
        *,
        algorithm: str = "auto",
    ) -> None:
        self._reference = as_point_cloud(reference, "reference")
        if algorithm not in {"auto", "bvh", "bruteforce"}:
            raise ValueError("algorithm must be 'auto', 'bvh', or 'bruteforce'")
        if algorithm == "bvh" and self._reference.batch_size != 1:
            raise ValueError("the bvh backend currently supports one reference cloud")
        self._algorithm = (
            "bvh"
            if algorithm == "auto" and self._reference.batch_size == 1
            else "bruteforce" if algorithm == "auto" else algorithm
        )
        self._backend = (
            BvhBackend(self._reference)
            if self._algorithm == "bvh"
            else PackedBackend(self._reference)
        )

    @property
    def reference(self) -> PointCloud:
        return self._reference

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def _validate_queries(self, queries: PointCloudLike) -> PointCloud:
        cloud = as_point_cloud(queries, "queries")
        check_cloud_pair(cloud, self._reference, "queries", "reference")
        return cloud

    def _search(
        self,
        queries: PointCloudLike,
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

        indices, _ = self._backend.knn(query_cloud, k, float(radius))
        indices = indices.to(torch.int64)
        backend_valid = indices >= 0
        if self._reference.points.shape[0] == 0:
            return Neighbors(
                indices=indices,
                distances2=torch.full(
                    indices.shape,
                    math.inf,
                    dtype=query_cloud.dtype,
                    device=query_cloud.device,
                ),
                valid=backend_valid,
            )
        gathered = self._reference.points[indices.clamp(min=0)]
        distances2 = (query_cloud.points[:, None] - gathered).square().sum(dim=-1)
        valid = backend_valid & (distances2 <= radius * radius)
        indices = indices.masked_fill(~valid, -1)
        distances2 = distances2.masked_fill(~valid, math.inf)
        return Neighbors(indices=indices, distances2=distances2, valid=valid)

    def knn(self, queries: PointCloudLike, k: int) -> Neighbors:
        """Return the ``k`` nearest neighbors for every query."""
        return self._search(queries, k=k, radius=math.inf)

    def radius(
        self,
        queries: PointCloudLike,
        radius: float,
        *,
        max_neighbors: int = 64,
    ) -> Neighbors:
        """Return up to ``max_neighbors`` within ``radius``."""
        return self._search(queries, k=max_neighbors, radius=float(radius))

    def hybrid(
        self,
        queries: PointCloudLike,
        radius: float,
        k: int,
    ) -> Neighbors:
        """Return up to ``k`` nearest neighbors within ``radius``."""
        return self._search(queries, k=k, radius=float(radius))


def knn(
    reference: PointCloudLike,
    queries: PointCloudLike,
    k: int,
    *,
    algorithm: str = "auto",
) -> Neighbors:
    return NeighborIndex(reference, algorithm=algorithm).knn(queries, k)


def radius_neighbors(
    reference: PointCloudLike,
    queries: PointCloudLike,
    radius: float,
    *,
    max_neighbors: int = 64,
    algorithm: str = "auto",
) -> Neighbors:
    return NeighborIndex(reference, algorithm=algorithm).radius(
        queries, radius, max_neighbors=max_neighbors
    )


def hybrid_neighbors(
    reference: PointCloudLike,
    queries: PointCloudLike,
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
