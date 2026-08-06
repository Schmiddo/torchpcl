"""Packed normal estimation using local covariance."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ._backend import symmetric_eigh_3x3 as _symmetric_eigh_3x3
from .cloud import PointCloud, PointCloudLike, as_point_cloud, batch_ids
from .local_geometry import _moments_from_neighbors
from .neighbors import NeighborIndex


@dataclass(frozen=True, eq=False)
class NormalResult:
    normals: torch.Tensor
    valid: torch.Tensor
    curvature: torch.Tensor


def _per_cloud_vector(
    value: torch.Tensor,
    cloud: PointCloud,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape not in ((3,), (cloud.batch_size, 3)):
        raise ValueError(
            f"{name} must have shape (3,) or ({cloud.batch_size}, 3), "
            f"got {tuple(value.shape)}"
        )
    if value.dtype != cloud.dtype:
        raise ValueError(f"{name} and cloud points must have the same dtype")
    if value.device != cloud.device:
        raise ValueError(f"{name} and cloud points must be on the same device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value.expand(cloud.batch_size, -1) if value.ndim == 1 else value


def _require_normals(cloud: PointCloud) -> torch.Tensor:
    if not isinstance(cloud, PointCloud):
        raise TypeError("cloud must be a PointCloud with attached normals")
    if cloud.normals is None:
        raise ValueError("cloud must have normals attached")
    return cloud.normals


def orient_normals_toward_viewpoint(
    cloud: PointCloud,
    viewpoint: torch.Tensor,
    *,
    toward: bool = True,
) -> PointCloud:
    """Flip attached normals toward or away from per-cloud viewpoints.

    Points exactly at their viewpoint, and normals perpendicular to the view
    direction, retain their original sign.
    """
    normals = _require_normals(cloud)
    if not isinstance(toward, bool):
        raise TypeError("toward must be a bool")
    viewpoints = _per_cloud_vector(viewpoint, cloud, "viewpoint")
    ids = batch_ids(cloud.offsets, cloud.points.shape[0])
    view_directions = viewpoints[ids] - cloud.points
    dots = (normals * view_directions).sum(dim=1)
    flip = dots < 0 if toward else dots > 0
    return replace(cloud, normals=torch.where(flip[:, None], -normals, normals))


def orient_normals_to_direction(
    cloud: PointCloud,
    direction: torch.Tensor,
    *,
    align: bool = True,
) -> PointCloud:
    """Flip attached normals to align with or oppose per-cloud directions."""
    normals = _require_normals(cloud)
    if not isinstance(align, bool):
        raise TypeError("align must be a bool")
    directions = _per_cloud_vector(direction, cloud, "direction")
    if bool(torch.any(~torch.any(directions != 0, dim=1))):
        raise ValueError("direction must be nonzero for every cloud")
    ids = batch_ids(cloud.offsets, cloud.points.shape[0])
    dots = (normals * directions[ids]).sum(dim=1)
    flip = dots < 0 if align else dots > 0
    return replace(cloud, normals=torch.where(flip[:, None], -normals, normals))


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


__all__ = [
    "NormalResult",
    "estimate_normals",
    "orient_normals_to_direction",
    "orient_normals_toward_viewpoint",
]
