"""Mask-first point-cloud validity checks and outlier detection."""

from __future__ import annotations

import math

import torch

from .cloud import PointCloudLike, as_point_cloud, batch_ids
from .neighbors import NeighborIndex


@torch.no_grad()
def radius_outliers(
    cloud: PointCloudLike,
    radius: float,
    k: int,
) -> torch.Tensor:
    """Classify points by inclusive-radius support, including the point itself."""
    if not 2 <= k <= 64:
        raise ValueError("k must be in [2, 64]")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")

    packed = as_point_cloud(cloud)
    index = NeighborIndex(packed)
    neighbors = index.hybrid(packed, radius, k)
    radius_valid = neighbors.valid & (neighbors.distances2 <= radius * radius)
    counts = radius_valid.sum(dim=1)
    return counts < k


def _segmented_sample_statistics(
    values: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-segment mean and sample std, ignoring non-finite rows."""
    batch_count = offsets.shape[0] - 1
    ids = batch_ids(offsets, values.shape[0])
    finite = torch.isfinite(values)
    counts = torch.zeros(batch_count, dtype=torch.int64, device=values.device)
    counts.scatter_add_(0, ids, finite.to(torch.int64))

    sums = torch.zeros(batch_count, dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, ids, torch.where(finite, values, 0))
    means = sums / counts.clamp(min=1).to(values.dtype)
    means = means.masked_fill(counts == 0, math.nan)

    centered2 = torch.where(finite, (values - means[ids]).square(), 0)
    squared_sums = torch.zeros_like(sums)
    squared_sums.scatter_add_(0, ids, centered2)
    std = torch.sqrt(squared_sums / (counts - 1).clamp(min=1).to(values.dtype))
    std = std.masked_fill(counts == 0, math.nan)
    return means, std


@torch.no_grad()
def statistical_outliers(
    cloud: PointCloudLike,
    k: int,
    std_ratio: float = 2.0,
) -> torch.Tensor:
    """Classify points from mean k-NN distance, including the point itself."""
    if not 2 <= k <= 64:
        raise ValueError("k must be in [2, 64]")
    if not math.isfinite(std_ratio) or std_ratio < 0:
        raise ValueError("std_ratio must be finite and nonnegative")

    packed = as_point_cloud(cloud)
    index = NeighborIndex(packed)
    neighbors = index.knn(packed, k)
    # count should be k except for point clouds with less than k points
    # not sure if we need to keep this
    counts = neighbors.valid.sum(dim=1)
    distances = neighbors.distances2.clamp_min(0).sqrt()
    distance_sums = torch.where(neighbors.valid, distances, 0).sum(dim=1)
    # a point is included in its neighborhood, so count is always at least one
    means = distance_sums / counts.to(packed.dtype)

    # cloud mean/std + thresholds are shape (B,)
    cloud_mean, cloud_std = _segmented_sample_statistics(means, packed.offsets)
    thresholds = cloud_mean + float(std_ratio) * cloud_std
    ids = batch_ids(packed.offsets, packed.points.shape[0])
    outlier_mask = ~torch.isfinite(means) | (means > thresholds[ids])
    return outlier_mask


__all__ = [
    "radius_outliers",
    "statistical_outliers",
]
