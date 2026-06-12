"""Point cloud preprocessing: normal estimation and voxel downsampling."""

import math

import torch

from .search import NearestNeighborSearch
from .transforms import _check_points


def _smallest_eigenvector_3x3(cov: torch.Tensor) -> torch.Tensor:
    """Unit eigenvector of the smallest eigenvalue per (N, 3, 3) symmetric
    matrix, computed analytically (batched, float64 internally).

    Avoids torch.linalg.eigh, whose batched CUDA path allocates an
    excessive per-matrix cuSOLVER workspace. Uses the trigonometric
    eigenvalue formula for symmetric 3x3 matrices, then takes the
    largest cross product of rows of (A - lambda_min I).
    """
    a = cov.to(torch.float64)
    n = len(a)
    eye = torch.eye(3, dtype=a.dtype, device=a.device)

    trace_third = a.diagonal(dim1=1, dim2=2).sum(-1) / 3.0
    b = a - trace_third[:, None, None] * eye
    p = ((b * b).sum(dim=(1, 2)) / 6.0).sqrt()
    p_safe = p.clamp(min=torch.finfo(a.dtype).tiny)
    bn = b / p_safe[:, None, None]
    # det of the normalized deviator, written out to stay solver-free.
    det = (
        bn[:, 0, 0] * (bn[:, 1, 1] * bn[:, 2, 2] - bn[:, 1, 2] * bn[:, 2, 1])
        - bn[:, 0, 1] * (bn[:, 1, 0] * bn[:, 2, 2] - bn[:, 1, 2] * bn[:, 2, 0])
        + bn[:, 0, 2] * (bn[:, 1, 0] * bn[:, 2, 1] - bn[:, 1, 1] * bn[:, 2, 0])
    )
    phi = torch.acos((det / 2.0).clamp(-1.0, 1.0)) / 3.0
    smallest = trace_third + 2.0 * p * torch.cos(phi + 2.0 * math.pi / 3.0)

    m = a - smallest[:, None, None] * eye
    crosses = torch.stack(
        [
            torch.linalg.cross(m[:, 0], m[:, 1]),
            torch.linalg.cross(m[:, 0], m[:, 2]),
            torch.linalg.cross(m[:, 1], m[:, 2]),
        ],
        dim=1,
    )
    norms = crosses.norm(dim=2)
    best = crosses[torch.arange(n, device=a.device), norms.argmax(dim=1)]
    return best / best.norm(dim=1, keepdim=True).clamp(min=torch.finfo(a.dtype).tiny)


def estimate_normals(
    points: torch.Tensor,
    radius: float | None = None,
    *,
    k: int = 30,
    viewpoint: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate per-point normals from local covariance.

    For each point, up to ``k`` nearest neighbors are gathered (within
    ``radius`` if given, unbounded true k-NN otherwise) and the normal is
    the eigenvector of the neighborhood covariance with the smallest
    eigenvalue.

    Args:
        points: (N, 3) point cloud.
        radius: Optional neighbor search radius; ``None`` means unbounded
            k-NN.
        k: Maximum number of neighbors per point (capped at 64).
        viewpoint: Optional (3,) location; normals are flipped to point
            towards it. Without it the sign of each normal is arbitrary
            (which is fine for point-to-plane ICP).

    Returns:
        (N, 3) unit normals in the input dtype. Points with fewer than 3
        neighbors get a zero normal.
    """
    _check_points(points, "points")
    if radius is not None and radius <= 0:
        raise ValueError("radius must be positive")
    if k < 3:
        raise ValueError("k must be at least 3")

    nns = NearestNeighborSearch(points, math.inf if radius is None else radius)
    indices, _ = nns.knn_query(points, k)
    valid = indices >= 0
    counts = valid.sum(dim=1)

    neighbors = points[indices.clamp(min=0)]  # (N, k, 3)
    mask = valid.unsqueeze(-1).to(points.dtype)
    denom = counts.clamp(min=1).unsqueeze(-1).to(points.dtype)
    mean = (neighbors * mask).sum(dim=1) / denom
    centered = (neighbors - mean.unsqueeze(1)) * mask
    cov = centered.transpose(1, 2) @ centered  # (N, 3, 3)

    normals = _smallest_eigenvector_3x3(cov).to(points.dtype)
    normals = torch.where((counts >= 3).unsqueeze(-1), normals, torch.zeros_like(normals))

    if viewpoint is not None:
        to_viewpoint = viewpoint.to(points.dtype).to(points.device) - points
        flip = (normals * to_viewpoint).sum(dim=-1, keepdim=True) < 0
        normals = torch.where(flip, -normals, normals)
    return normals


def voxel_downsample(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Downsample by averaging all points that fall in the same voxel.

    Returns the per-voxel mean points (in arbitrary voxel order), in the
    input dtype. Matches Open3D/small_gicp voxel-grid downsampling.
    """
    _check_points(points, "points")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    coords = torch.floor(points / voxel_size).to(torch.int64)
    coords = coords - coords.amin(dim=0)
    if int(coords.amax()) >= 1 << 21:
        raise ValueError(
            "point cloud spans more than 2^21 voxels per axis; "
            "use a larger voxel_size"
        )
    keys = (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]
    _, inverse = torch.unique(keys, return_inverse=True)
    num_voxels = int(inverse.max()) + 1

    sums = torch.zeros(num_voxels, 3, dtype=points.dtype, device=points.device)
    sums.index_add_(0, inverse, points)
    counts = torch.bincount(inverse, minlength=num_voxels).to(points.dtype)
    return sums / counts.unsqueeze(1)
