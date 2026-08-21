"""Point-aligned local feature descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .cloud import PointCloud
from .neighbors import NeighborIndex


@dataclass(frozen=True, eq=False)
class FPFHResult:
    """Fast point feature histograms and per-point diagnostics."""

    descriptors: torch.Tensor
    valid: torch.Tensor
    neighbor_limit_reached: torch.Tensor


def _normalized_normals(cloud: PointCloud) -> tuple[torch.Tensor, torch.Tensor]:
    assert cloud.normals is not None
    finite = torch.isfinite(cloud.normals).all(dim=1)
    norms = torch.linalg.vector_norm(cloud.normals, dim=1)
    valid = finite & (norms > torch.finfo(cloud.dtype).tiny)
    safe_norms = torch.where(valid, norms, torch.ones_like(norms))
    normalized = cloud.normals / safe_norms[:, None]
    return torch.where(valid[:, None], normalized, 0), valid


def _pair_features(
    points: torch.Tensor,
    normals: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Open3D/PCL-style (theta, alpha, phi) pair features."""
    gathered_points = points[indices]
    gathered_normals = normals[indices]
    displacement = gathered_points - points[:, None, :]
    distance = torch.linalg.vector_norm(displacement, dim=2)
    nonzero = distance > torch.finfo(points.dtype).tiny
    safe_distance = torch.where(nonzero, distance, torch.ones_like(distance))

    angle1 = (normals[:, None, :] * displacement).sum(dim=2) / safe_distance
    angle2 = (gathered_normals * displacement).sum(dim=2) / safe_distance
    angle1 = angle1.clamp(-1, 1)
    angle2 = angle2.clamp(-1, 1)
    swap = angle1.abs() < angle2.abs()

    source_normal = torch.where(
        swap[:, :, None], gathered_normals, normals[:, None, :]
    )
    target_normal = torch.where(
        swap[:, :, None], normals[:, None, :], gathered_normals
    )
    oriented_displacement = torch.where(
        swap[:, :, None], -displacement, displacement
    )
    phi = torch.where(swap, -angle2, angle1)

    v = torch.linalg.cross(oriented_displacement, source_normal, dim=2)
    v_norm = torch.linalg.vector_norm(v, dim=2)
    frame_valid = nonzero & (v_norm > torch.finfo(points.dtype).tiny)
    safe_v_norm = torch.where(frame_valid, v_norm, torch.ones_like(v_norm))
    v = v / safe_v_norm[:, :, None]
    w = torch.linalg.cross(source_normal, v, dim=2)

    alpha = (v * target_normal).sum(dim=2).clamp(-1, 1)
    theta = torch.atan2(
        (w * target_normal).sum(dim=2),
        (source_normal * target_normal).sum(dim=2),
    )
    features = torch.stack((theta, alpha, phi), dim=2)
    features = torch.where(frame_valid[:, :, None], features, 0)
    return features, nonzero


def _feature_bins(features: torch.Tensor) -> torch.Tensor:
    theta = torch.floor(11 * (features[:, :, 0] + math.pi) / (2 * math.pi))
    alpha = torch.floor(11 * (features[:, :, 1] + 1) * 0.5)
    phi = torch.floor(11 * (features[:, :, 2] + 1) * 0.5)
    bins = torch.stack((theta, alpha, phi), dim=2).to(torch.int64)
    bins.clamp_(0, 10)
    blocks = torch.tensor([0, 11, 22], dtype=torch.int64, device=features.device)
    return bins + blocks


@torch.no_grad()
def fpfh(
    cloud: PointCloud,
    radius: float,
    *,
    max_neighbors: int = 64,
) -> FPFHResult:
    """Compute 33-value FPFH descriptors using bounded hybrid neighborhoods.

    Normals must already be attached and consistently oriented. Invalid rows
    are zero. ``neighbor_limit_reached`` identifies rows for which the bounded
    search filled every slot and may therefore have omitted radius neighbors.
    """
    if not isinstance(cloud, PointCloud):
        raise TypeError("cloud must be a PointCloud with attached normals")
    if cloud.normals is None:
        raise ValueError("cloud must have normals attached")
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not isinstance(max_neighbors, int) or not 2 <= max_neighbors <= 64:
        raise ValueError("max_neighbors must be in [2, 64]")

    point_count = cloud.points.shape[0]
    if point_count == 0:
        return FPFHResult(
            descriptors=cloud.points.new_empty((0, 33)),
            valid=torch.empty(0, dtype=torch.bool, device=cloud.device),
            neighbor_limit_reached=torch.empty(
                0, dtype=torch.bool, device=cloud.device
            ),
        )

    neighbors = NeighborIndex(cloud).hybrid(cloud, float(radius), max_neighbors)
    indices = neighbors.indices.clamp(min=0)
    query_rows = torch.arange(point_count, device=cloud.device)[:, None]
    normals, normal_valid = _normalized_normals(cloud)
    edge_valid = (
        neighbors.valid
        & (indices != query_rows)
        & normal_valid[:, None]
        & normal_valid[indices]
    )

    pair_features, nonzero_distance = _pair_features(
        cloud.points, normals, indices
    )
    histogram_indices = _feature_bins(pair_features).reshape(
        point_count, max_neighbors * 3
    )
    edge_counts = edge_valid.sum(dim=1)
    increments = torch.where(
        edge_counts > 0,
        100.0 / edge_counts.clamp(min=1).to(cloud.dtype),
        torch.zeros(point_count, dtype=cloud.dtype, device=cloud.device),
    )
    histogram_values = (
        edge_valid[:, :, None].expand(-1, -1, 3).reshape(
            point_count, max_neighbors * 3
        )
        * increments[:, None]
    )
    spfh = torch.zeros(
        (point_count, 33), dtype=cloud.dtype, device=cloud.device
    )
    spfh.scatter_add_(1, histogram_indices, histogram_values)
    spfh_valid = edge_counts > 0

    aggregation_valid = (
        edge_valid & nonzero_distance & spfh_valid[indices]
    )
    weights = torch.where(
        aggregation_valid,
        neighbors.distances2.clamp_min(torch.finfo(cloud.dtype).tiny).reciprocal(),
        0,
    )
    neighbor_spfh = spfh[indices]
    aggregated = (neighbor_spfh * weights[:, :, None]).sum(dim=1)
    blocks = aggregated.reshape(point_count, 3, 11)
    block_sums = blocks.sum(dim=2, keepdim=True)
    blocks = torch.where(
        block_sums > 0,
        blocks * (100.0 / block_sums.clamp_min(torch.finfo(cloud.dtype).tiny)),
        0,
    )
    aggregated = blocks.reshape(point_count, 33)

    aggregation_counts = aggregation_valid.sum(dim=1)
    valid = spfh_valid & (aggregation_counts > 0)
    descriptors = torch.where(valid[:, None], spfh + aggregated, 0)
    return FPFHResult(
        descriptors=descriptors,
        valid=valid,
        neighbor_limit_reached=neighbors.valid.sum(dim=1) == max_neighbors,
    )


__all__ = ["FPFHResult", "fpfh"]
