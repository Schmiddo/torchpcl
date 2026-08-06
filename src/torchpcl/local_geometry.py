"""Differentiable local moments for packed point clouds."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .cloud import PointCloudLike, as_point_cloud
from .neighbors import NeighborIndex, Neighbors


@dataclass(frozen=True, eq=False)
class CovarianceResult:
    """Per-point neighborhood means and covariance matrices."""

    means: torch.Tensor
    covariances: torch.Tensor
    counts: torch.Tensor
    valid: torch.Tensor


def _moments_from_neighbors(
    points: torch.Tensor,
    neighbors: Neighbors,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return means, centered scatter matrices, and valid-slot counts."""
    counts = neighbors.valid.sum(dim=1)
    gathered = points[neighbors.indices.clamp(min=0)]
    weights = neighbors.valid.unsqueeze(-1).to(points.dtype)
    means = (gathered * weights).sum(dim=1)
    means = means / counts.clamp(min=1).to(points.dtype)[:, None]
    centered = (gathered - means[:, None]) * weights
    scatter = centered.transpose(1, 2) @ centered
    return means, scatter, counts


def _validate_neighbors(points: torch.Tensor, neighbors: Neighbors) -> None:
    if not isinstance(neighbors, Neighbors):
        raise TypeError("neighbors must be a Neighbors result")
    shape = neighbors.indices.shape
    if neighbors.indices.ndim != 2 or shape[0] != points.shape[0]:
        raise ValueError(
            "neighbors must have one fixed-width row per packed cloud point"
        )
    if neighbors.distances2.shape != shape or neighbors.valid.shape != shape:
        raise ValueError("neighbor fields must have identical two-dimensional shapes")
    if neighbors.indices.dtype != torch.int64:
        raise ValueError("neighbor indices must have dtype int64")
    if neighbors.valid.dtype != torch.bool:
        raise ValueError("neighbor validity must have dtype bool")
    if neighbors.distances2.dtype != points.dtype:
        raise ValueError("neighbor distances and cloud points must have the same dtype")
    fields = (neighbors.indices, neighbors.distances2, neighbors.valid)
    if any(field.device != points.device for field in fields):
        raise ValueError("neighbor fields and cloud points must be on the same device")
    invalid_index = (neighbors.indices < -1) | (neighbors.indices >= points.shape[0])
    invalid_index |= neighbors.valid & (neighbors.indices < 0)
    if bool(invalid_index.any()):
        raise ValueError("neighbor indices must be -1 or refer to packed cloud rows")


def covariance_from_neighbors(
    cloud: PointCloudLike,
    neighbors: Neighbors,
    *,
    correction: int = 1,
) -> CovarianceResult:
    """Compute local covariances from a fixed-width self-neighborhood.

    Neighbor identities are discrete, while means and covariances retain
    ordinary gradients to every gathered point row.
    """
    if not isinstance(correction, int) or isinstance(correction, bool):
        raise TypeError("correction must be an integer")
    if correction not in (0, 1):
        raise ValueError("correction must be 0 or 1")

    packed = as_point_cloud(cloud)
    if not bool(torch.isfinite(packed.points).all()):
        raise ValueError("cloud points must be finite")
    _validate_neighbors(packed.points, neighbors)

    means, scatter, counts = _moments_from_neighbors(packed.points, neighbors)
    valid = (counts >= 3) & (counts > correction)
    denominator = (counts - correction).clamp(min=1).to(packed.dtype)
    covariances = scatter / denominator[:, None, None]
    means = torch.where(valid[:, None], means, torch.zeros_like(means))
    covariances = torch.where(
        valid[:, None, None], covariances, torch.zeros_like(covariances)
    )
    return CovarianceResult(means, covariances, counts, valid)


def estimate_covariances(
    cloud: PointCloudLike,
    radius: float | None = None,
    *,
    k: int = 30,
    index: NeighborIndex | None = None,
    correction: int = 1,
) -> CovarianceResult:
    """Estimate bounded k-NN or hybrid covariances around every point."""
    if not isinstance(k, int) or isinstance(k, bool) or not 3 <= k <= 64:
        raise ValueError("k must be an integer in [3, 64]")
    if radius is not None and (not math.isfinite(radius) or radius <= 0):
        raise ValueError("radius must be finite and positive")

    packed = as_point_cloud(cloud)
    if not bool(torch.isfinite(packed.points).all()):
        raise ValueError("cloud points must be finite")
    if index is None:
        index = NeighborIndex(packed)
    neighbors = (
        index.knn(packed, k)
        if radius is None
        else index.hybrid(packed, float(radius), k)
    )
    return covariance_from_neighbors(packed, neighbors, correction=correction)


__all__ = [
    "CovarianceResult",
    "covariance_from_neighbors",
    "estimate_covariances",
]
