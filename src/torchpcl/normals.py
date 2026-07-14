"""Packed normal estimation using local covariance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .cloud import PointCloud, as_cloud, batch_ids
from .neighbors import NeighborIndex


@dataclass(frozen=True, eq=False)
class NormalResult:
    normals: torch.Tensor
    valid: torch.Tensor
    curvature: torch.Tensor | None = None


@torch.no_grad()
def estimate_normals(
    cloud: torch.Tensor | PointCloud,
    radius: float | None = None,
    *,
    k: int = 30,
    viewpoint: torch.Tensor | None = None,
    orient: str = "none",
    include_self: bool = True,
    return_curvature: bool = False,
    index: NeighborIndex | None = None,
) -> NormalResult:
    """Estimate normals from the smallest local covariance eigenvector.

    The query point is included by default. Normals with fewer than three valid
    neighbors are zero and marked invalid. This operation is currently
    inference-only.
    """
    packed = as_cloud(cloud)
    if radius is not None and radius <= 0:
        raise ValueError("radius must be positive")
    if k < 3 or k > 64:
        raise ValueError("k must be in [3, 64]")
    if not include_self and k == 64:
        raise ValueError("include_self=False currently requires k <= 63")
    if orient not in {"none", "viewpoint"}:
        raise ValueError("orient must be 'none' or 'viewpoint'")
    if viewpoint is not None:
        orient = "viewpoint"
    if orient == "viewpoint" and viewpoint is None:
        raise ValueError("orient='viewpoint' requires viewpoint")

    if index is None:
        index = NeighborIndex(packed)
    elif index.reference.points.data_ptr() != packed.points.data_ptr():
        raise ValueError("index must have been built for cloud")

    search_k = k if include_self else k + 1
    if radius is None:
        neighbors = index.knn(packed, search_k)
    else:
        neighbors = index.hybrid(packed, radius, search_k)

    indices = neighbors.indices
    valid = neighbors.valid
    if not include_self:
        rows = torch.arange(packed.points.shape[0], device=packed.device)[:, None]
        keep = valid & (indices != rows)
        sortable = neighbors.distances2.masked_fill(~keep, math.inf)
        order = sortable.argsort(dim=1)[:, :k]
        indices = indices.gather(1, order)
        valid = keep.gather(1, order)

    counts = valid.sum(dim=1)
    gathered = packed.points[indices.clamp(min=0)]
    weights = valid.unsqueeze(-1).to(packed.dtype)
    means = (gathered * weights).sum(dim=1)
    means = means / counts.clamp(min=1).to(packed.dtype)[:, None]
    centered = (gathered - means[:, None]) * weights
    covariance = centered.transpose(1, 2) @ centered

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    normal_valid = counts >= 3
    normals = torch.where(normal_valid[:, None], normals, torch.zeros_like(normals))

    if orient == "viewpoint":
        assert viewpoint is not None
        viewpoint = viewpoint.to(device=packed.device, dtype=packed.dtype)
        if viewpoint.shape == (3,):
            viewpoint = viewpoint.expand(packed.batch_size, -1)
        elif viewpoint.shape != (packed.batch_size, 3):
            raise ValueError(
                f"viewpoint must have shape (3,) or ({packed.batch_size}, 3)"
            )
        ids = batch_ids(packed.offsets, packed.points.shape[0])
        toward_viewpoint = viewpoint[ids] - packed.points
        flip = (normals * toward_viewpoint).sum(dim=1) < 0
        normals = torch.where(flip[:, None], -normals, normals)

    curvature = None
    if return_curvature:
        denominator = eigenvalues.sum(dim=1).clamp_min(
            torch.finfo(packed.dtype).tiny
        )
        curvature = eigenvalues[:, 0] / denominator
        curvature = torch.where(
            normal_valid, curvature, torch.zeros_like(curvature)
        )
    return NormalResult(normals=normals, valid=normal_valid, curvature=curvature)


__all__ = ["NormalResult", "estimate_normals"]
