"""Packed normal estimation using local covariance."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._backend import symmetric_eigh_3x3 as _symmetric_eigh_3x3
from .cloud import PointCloudLike, as_point_cloud
from .local_geometry import _moments_from_neighbors
from .neighbors import NeighborIndex


@dataclass(frozen=True, eq=False)
class NormalResult:
    normals: torch.Tensor
    valid: torch.Tensor
    curvature: torch.Tensor


@torch.no_grad()
def estimate_normals(
    cloud: PointCloudLike,
    radius: float | None = None,
    *,
    k: int = 30,
    index: NeighborIndex | None = None,
) -> NormalResult:
    """Estimate normals from the smallest local covariance eigenvector.

    The neighborhood always includes the query point. If the cloud has
    normals attached, estimated normals are sign-aligned to them; otherwise
    signs are arbitrary and orientation is left to the caller. Normals with
    fewer than three valid neighbors are zero and marked invalid, as is their
    curvature. This operation is currently inference-only.
    """
    packed = as_point_cloud(cloud)
    if radius is not None and radius <= 0:
        raise ValueError("radius must be positive")
    if k < 3 or k > 64:
        raise ValueError("k must be in [3, 64]")

    if index is None:
        index = NeighborIndex(packed)

    if radius is None:
        neighbors = index.knn(packed, k)
    else:
        neighbors = index.hybrid(packed, radius, k)

    _, covariance, counts = _moments_from_neighbors(packed.points, neighbors)

    eigenvalues, normals = _symmetric_eigh_3x3(covariance)
    normal_valid = counts >= 3
    normals = torch.where(normal_valid[:, None], normals, torch.zeros_like(normals))

    if packed.normals is not None:
        flip = (normals * packed.normals).sum(dim=1) < 0
        normals = torch.where(flip[:, None], -normals, normals)

    denominator = eigenvalues.sum(dim=1).clamp_min(torch.finfo(packed.dtype).tiny)
    curvature = eigenvalues[:, 0] / denominator
    curvature = torch.where(normal_valid, curvature, torch.zeros_like(curvature))
    return NormalResult(normals=normals, valid=normal_valid, curvature=curvature)


__all__ = ["NormalResult", "estimate_normals"]
