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


def covariance_from_neighbors(
    cloud: PointCloudLike,
    neighbors: Neighbors,
) -> CovarianceResult:
    """Compute local covariances from a fixed-width self-neighborhood.

    Neighbor identities are discrete, while means and covariances retain
    ordinary gradients to every gathered point row.
    """

    packed = as_point_cloud(cloud)

    means, scatter, counts = _moments_from_neighbors(packed.points, neighbors)
    valid = (counts >= 3)
    denominator = (counts - 1).clamp(min=1).to(packed.dtype)
    covariances = scatter / denominator[:, None, None]
    means = torch.where(valid[:, None], means, torch.zeros_like(means))
    covariances = torch.where(
        valid[:, None, None], covariances, torch.zeros_like(covariances)
    )
    return CovarianceResult(means, covariances, counts, valid)


def estimate_covariances(
    cloud: PointCloudLike,
    *,
    radius: float = math.inf,
    k: int = 30,
) -> CovarianceResult:
    """Estimate bounded covariances around every point."""
    if not isinstance(k, int) or isinstance(k, bool) or not 3 <= k <= 64:
        raise ValueError("k must be an integer in [3, 64]")
    if radius <= 0:
        raise ValueError("radius must be positive")

    packed = as_point_cloud(cloud)
    index = NeighborIndex(packed)
    neighbors = index.hybrid(packed, radius, k)
    return covariance_from_neighbors(packed, neighbors)


__all__ = [
    "CovarianceResult",
    "estimate_covariances",
]
