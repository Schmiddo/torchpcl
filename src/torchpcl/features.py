"""Point-aligned local feature descriptors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._backend import fpfh as _fpfh
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
    normals, normal_valid = _normalized_normals(cloud)
    descriptors, valid = _fpfh(
        cloud.points, normals, normal_valid, neighbors.indices, neighbors.distances2
    )
    return FPFHResult(
        descriptors=descriptors,
        valid=valid,
        neighbor_limit_reached=neighbors.valid.sum(dim=1) == max_neighbors,
    )


__all__ = ["FPFHResult", "fpfh"]
