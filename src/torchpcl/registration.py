"""Single-scale ICP registration loop and evaluation."""

import math

import torch

from .estimation import PointToPoint, TransformationEstimation
from .search import NearestNeighborSearch
from .transforms import _check_points, transform_points
from .types import ICPConvergenceCriteria, RegistrationResult


def _make_nns(points: torch.Tensor, radius: float, backend: str):
    if backend == "warp":
        return NearestNeighborSearch(points, radius)
    if backend == "cubql":
        from .search_cubql import CuBQLNearestNeighborSearch

        return CuBQLNearestNeighborSearch(points, radius)
    raise ValueError(f"unknown backend {backend!r}, expected 'warp' or 'cubql'")


def _evaluate(
    nns: NearestNeighborSearch,
    points: torch.Tensor,
    transformation: torch.Tensor,
) -> tuple[RegistrationResult, torch.Tensor]:
    """Compute correspondences and metrics for the given transformation.

    Returns the result and the inlier mask over the source points.
    """
    indices, dist2 = nns.query(points)
    mask = indices >= 0
    # Unmatched slots hold radius^2, so zero them before summing; one
    # stacked transfer keeps this at a single host sync per iteration.
    stats = torch.stack(
        [mask.sum().to(torch.float64), (dist2.to(torch.float64) * mask).sum()]
    ).cpu()
    num_inliers = int(stats[0])
    fitness = num_inliers / len(points)
    inlier_rmse = math.sqrt(float(stats[1]) / num_inliers) if num_inliers > 0 else 0.0
    result = RegistrationResult(
        transformation=transformation.clone(),
        correspondences=indices,
        fitness=fitness,
        inlier_rmse=inlier_rmse,
    )
    return result, mask


def _validate_inputs(
    source: torch.Tensor,
    target: torch.Tensor,
    max_correspondence_distance: float,
    estimation: TransformationEstimation,
    target_normals: torch.Tensor | None,
) -> None:
    _check_points(source, "source")
    _check_points(target, "target")
    if len(source) == 0 or len(target) == 0:
        raise ValueError("source and target must be non-empty")
    if source.device != target.device:
        raise ValueError(
            f"source and target must be on the same device, "
            f"got {source.device} and {target.device}"
        )
    if max_correspondence_distance <= 0:
        raise ValueError("max_correspondence_distance must be positive")
    if estimation.requires_normals:
        if target_normals is None:
            raise ValueError(
                f"{type(estimation).__name__} requires target_normals"
            )
        _check_points(target_normals, "target_normals")
        if target_normals.shape != target.shape:
            raise ValueError(
                "target_normals must have the same shape as target, "
                f"got {tuple(target_normals.shape)} and {tuple(target.shape)}"
            )
        if target_normals.device != target.device:
            raise ValueError("target_normals must be on the same device as target")


def icp(
    source: torch.Tensor,
    target: torch.Tensor,
    max_correspondence_distance: float,
    *,
    init: torch.Tensor | None = None,
    estimation: TransformationEstimation | None = None,
    criteria: ICPConvergenceCriteria = ICPConvergenceCriteria(),
    target_normals: torch.Tensor | None = None,
    backend: str = "warp",
) -> RegistrationResult:
    """Register source to target with single-scale ICP.

    Args:
        source: (N, 3) source points.
        target: (M, 3) target points.
        max_correspondence_distance: Correspondence search radius.
        init: Optional (4, 4) initial source-to-target transformation.
        estimation: Transformation estimation method (default PointToPoint).
        criteria: Convergence criteria.
        target_normals: (M, 3) target normals, required for PointToPlane.
        backend: Correspondence search backend, "warp" (default, CPU+CUDA)
            or "cubql" (experimental, CUDA-only, JIT-compiled).

    Note: if at any iteration no correspondences are found, the current
    transformation is kept and the result has converged=False, fitness=0
    (Open3D resets to identity in this case; we deliberately do not).
    """
    estimation = estimation if estimation is not None else PointToPoint()
    _validate_inputs(source, target, max_correspondence_distance, estimation, target_normals)

    device = source.device
    # Points are kept in the input precision; only the cumulative
    # transformation and the small solves (in estimation) are float64.
    source_pts = source.contiguous()
    target_pts = target.to(source.dtype).contiguous()
    normals_pts = (
        target_normals.to(source.dtype).contiguous() if target_normals is not None else None
    )
    if init is not None:
        transformation = init.to(device=device, dtype=torch.float64).clone()
    else:
        transformation = torch.eye(4, dtype=torch.float64, device=device)

    nns = _make_nns(target_pts, max_correspondence_distance, backend)
    current = transform_points(source_pts, transformation)

    prev_fitness = 0.0
    prev_inlier_rmse = 0.0
    converged = False
    num_iterations = 0
    for iteration in range(criteria.max_iteration):
        result, mask = _evaluate(nns, current, transformation)

        if result.fitness <= 0.0:
            result.converged = False
            result.num_iterations = num_iterations
            return result

        if (
            iteration > 0
            and abs(prev_fitness - result.fitness) < criteria.relative_fitness
            and abs(prev_inlier_rmse - result.inlier_rmse) < criteria.relative_rmse
        ):
            converged = True
            break
        prev_fitness = result.fitness
        prev_inlier_rmse = result.inlier_rmse

        inlier_indices = result.correspondences[mask]
        delta = estimation.compute_transformation(
            current[mask],
            target_pts[inlier_indices],
            normals_pts[inlier_indices] if normals_pts is not None else None,
        )
        transformation = delta @ transformation
        # Re-apply the full float64 transformation to the original points
        # rather than chaining deltas, so float32 rounding does not drift.
        current = transform_points(source_pts, transformation)
        num_iterations += 1

    # Recompute metrics for the final transformation (matches Open3D,
    # which re-evaluates after the iteration loop ends).
    result, _ = _evaluate(nns, current, transformation)
    result.converged = converged
    result.num_iterations = num_iterations
    return result


def evaluate_registration(
    source: torch.Tensor,
    target: torch.Tensor,
    max_correspondence_distance: float,
    transformation: torch.Tensor | None = None,
    *,
    backend: str = "warp",
) -> RegistrationResult:
    """Compute fitness/inlier RMSE of a transformation without iterating."""
    _check_points(source, "source")
    _check_points(target, "target")
    if len(source) == 0 or len(target) == 0:
        raise ValueError("source and target must be non-empty")
    if max_correspondence_distance <= 0:
        raise ValueError("max_correspondence_distance must be positive")

    source_pts = source.contiguous()
    target_pts = target.to(source.dtype).contiguous()
    if transformation is not None:
        transformation = transformation.to(device=source.device, dtype=torch.float64)
    else:
        transformation = torch.eye(4, dtype=torch.float64, device=source.device)

    nns = _make_nns(target_pts, max_correspondence_distance, backend)
    result, _ = _evaluate(nns, transform_points(source_pts, transformation), transformation)
    return result
