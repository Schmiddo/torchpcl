"""Batched single-scale iterative closest point registration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .alignment import _procrustes_packed
from .cloud import PointCloud, as_cloud, batch_ids
from .neighbors import NeighborIndex
from .transforms import transform
from .types import ICPResult, RegistrationMetrics


@dataclass(frozen=True, eq=False)
class _Evaluation:
    current: torch.Tensor
    target: torch.Tensor
    indices: torch.Tensor
    valid: torch.Tensor
    counts: torch.Tensor
    fitness: torch.Tensor
    rmse: torch.Tensor


def _segment_sum(
    values: torch.Tensor,
    ids: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if batch_size == 1:
        return values.sum(dim=0, keepdim=True)
    output = values.new_zeros((batch_size, *values.shape[1:]))
    output.index_add_(0, ids, values)
    return output


def _prepare_inputs(
    source: torch.Tensor | PointCloud,
    target: torch.Tensor | PointCloud,
    max_distance: float,
) -> tuple[PointCloud, PointCloud]:
    source_cloud = as_cloud(source, "source")
    target_cloud = as_cloud(target, "target")
    if source_cloud.batch_size != target_cloud.batch_size:
        raise ValueError("source and target must have the same batch size")
    if source_cloud.device != target_cloud.device:
        raise ValueError("source and target must be on the same device")
    if source_cloud.dtype != target_cloud.dtype:
        raise ValueError("source and target must have the same dtype")
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    empty = torch.any((source_cloud.lengths == 0) | (target_cloud.lengths == 0))
    if bool(empty):
        raise ValueError("source and target batches must be non-empty")
    return source_cloud, target_cloud


def _initial_transforms(
    cloud: PointCloud,
    init: torch.Tensor | None,
) -> torch.Tensor:
    if init is None:
        return torch.eye(4, dtype=cloud.dtype, device=cloud.device).repeat(
            cloud.batch_size, 1, 1
        )
    if not isinstance(init, torch.Tensor):
        raise TypeError("init must be a torch.Tensor")
    if init.shape == (4, 4):
        init = init.unsqueeze(0).expand(cloud.batch_size, -1, -1)
    elif init.shape != (cloud.batch_size, 4, 4):
        raise ValueError(
            f"init must have shape (4, 4) or ({cloud.batch_size}, 4, 4)"
        )
    return init.to(device=cloud.device, dtype=cloud.dtype).clone()


def _target_normals(
    target: PointCloud,
    normals: torch.Tensor | None,
    method: str,
) -> torch.Tensor | None:
    if normals is None:
        normals = target.normals
    if method == "point_to_plane" and normals is None:
        raise ValueError("point_to_plane requires target_normals")
    if normals is None:
        return None
    if not isinstance(normals, torch.Tensor) or normals.shape != target.points.shape:
        raise ValueError("target_normals must have shape (P, 3)")
    if normals.device != target.device or normals.dtype != target.dtype:
        raise ValueError("target_normals must match the target device and dtype")
    return normals


def _evaluate(
    source: PointCloud,
    target: PointCloud,
    index: NeighborIndex,
    transforms: torch.Tensor,
    max_distance: float,
    source_ids: torch.Tensor,
) -> _Evaluation:
    current_cloud = transform(source, transforms)
    assert isinstance(current_cloud, PointCloud)
    neighbors = index.hybrid(current_cloud, max_distance, 1)
    indices = neighbors.indices[:, 0]
    valid = neighbors.valid[:, 0]
    target_points = target.points[indices.clamp(min=0)]
    distances2 = neighbors.distances2[:, 0].masked_fill(~valid, 0)
    counts = _segment_sum(valid.to(torch.int64), source_ids, source.batch_size)
    squared_error = _segment_sum(distances2, source_ids, source.batch_size)
    fitness = counts.to(source.dtype) / source.lengths.to(source.dtype)
    rmse = torch.where(
        counts > 0,
        (squared_error / counts.clamp(min=1).to(source.dtype)).sqrt(),
        torch.zeros_like(squared_error),
    )
    return _Evaluation(
        current=current_cloud.points,
        target=target_points,
        indices=indices,
        valid=valid,
        counts=counts,
        fitness=fitness,
        rmse=rmse,
    )


def _robust_weights(
    residual_magnitude: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    ids: torch.Tensor,
    robust_kernel: str | None,
    robust_delta: float,
) -> torch.Tensor:
    weights = (valid & active[ids]).to(residual_magnitude.dtype)
    if robust_kernel == "huber":
        magnitude = residual_magnitude.clamp_min(
            torch.finfo(residual_magnitude.dtype).tiny
        )
        huber = torch.where(
            residual_magnitude <= robust_delta,
            torch.ones_like(magnitude),
            robust_delta / magnitude,
        )
        weights = weights * huber
    return weights


def _point_to_point_delta(
    evaluation: _Evaluation,
    active: torch.Tensor,
    ids: torch.Tensor,
    batch_size: int,
    robust_kernel: str | None,
    robust_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = (evaluation.current - evaluation.target).norm(dim=1)
    weights = _robust_weights(
        residual,
        evaluation.valid,
        active,
        ids,
        robust_kernel,
        robust_delta,
    )
    alignment, _ = _procrustes_packed(
        evaluation.current,
        evaluation.target,
        weights,
        ids,
        batch_size,
        estimate_scale=False,
    )
    delta = alignment.transforms
    solvable = active & (evaluation.counts >= 3) & torch.isfinite(delta).all(
        dim=(1, 2)
    )
    return delta, solvable


def _point_to_plane_delta(
    evaluation: _Evaluation,
    normals: torch.Tensor,
    active: torch.Tensor,
    ids: torch.Tensor,
    batch_size: int,
    robust_kernel: str | None,
    robust_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    matched_normals = normals[evaluation.indices.clamp(min=0)]
    residual = ((evaluation.current - evaluation.target) * matched_normals).sum(
        dim=1
    )
    weights = _robust_weights(
        residual.abs(),
        evaluation.valid,
        active,
        ids,
        robust_kernel,
        robust_delta,
    )
    jacobian = torch.cat(
        [torch.linalg.cross(evaluation.current, matched_normals), matched_normals],
        dim=1,
    )
    jtj = _segment_sum(
        jacobian[:, :, None] * jacobian[:, None, :] * weights[:, None, None],
        ids,
        batch_size,
    )
    jtr = _segment_sum(
        jacobian * residual[:, None] * weights[:, None], ids, batch_size
    )
    pose, info = torch.linalg.solve_ex(jtj, -jtr)
    delta = _poses_to_matrices(pose)
    solvable = (
        active
        & (evaluation.counts >= 6)
        & (info == 0)
        & torch.isfinite(delta).all(dim=(1, 2))
    )
    return delta, solvable


def _rigid_matrices(
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    upper = torch.cat([rotation, translation[:, :, None]], dim=2)
    bottom = rotation.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(
        rotation.shape[0], 1, 4
    )
    return torch.cat([upper, bottom], dim=1)


def _poses_to_matrices(pose: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = pose[:, 0], pose[:, 1], pose[:, 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    row0 = torch.stack(
        [cz * cy, -sz * cx + cz * sy * sx, sz * sx + cz * sy * cx], dim=1
    )
    row1 = torch.stack(
        [sz * cy, cz * cx + sz * sy * sx, -cz * sx + sz * sy * cx], dim=1
    )
    row2 = torch.stack([-sy, cy * sx, cy * cx], dim=1)
    rotation = torch.stack([row0, row1, row2], dim=1)
    return _rigid_matrices(rotation, pose[:, 3:])


@torch.no_grad()
def icp(
    source: torch.Tensor | PointCloud,
    target: torch.Tensor | PointCloud,
    max_distance: float,
    *,
    init: torch.Tensor | None = None,
    method: str = "point_to_point",
    target_normals: torch.Tensor | None = None,
    max_iterations: int = 30,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
    robust_kernel: str | None = None,
    robust_delta: float = 1.0,
    index: NeighborIndex | None = None,
) -> ICPResult:
    """Register packed source clouds to corresponding target clouds.

    Each batch entry converges or fails independently. Failed entries retain
    their last valid transform. Correspondences are internal and are not
    returned or retained in the result.
    """
    source_cloud, target_cloud = _prepare_inputs(source, target, max_distance)
    if method not in {"point_to_point", "point_to_plane"}:
        raise ValueError("method must be 'point_to_point' or 'point_to_plane'")
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if relative_fitness < 0 or relative_rmse < 0:
        raise ValueError("convergence thresholds must be nonnegative")
    if robust_kernel not in {None, "huber"}:
        raise ValueError("robust_kernel must be None or 'huber'")
    if robust_delta <= 0:
        raise ValueError("robust_delta must be positive")
    normals = _target_normals(target_cloud, target_normals, method)
    transforms = _initial_transforms(source_cloud, init)
    if index is None:
        index = NeighborIndex(target_cloud)
    elif index.reference.points.data_ptr() != target_cloud.points.data_ptr():
        raise ValueError("index must have been built for target")

    batch_size = source_cloud.batch_size
    source_ids = batch_ids(source_cloud.offsets, source_cloud.points.shape[0])
    active = torch.ones(batch_size, dtype=torch.bool, device=source_cloud.device)
    converged = torch.zeros_like(active)
    iterations = torch.zeros(
        batch_size, dtype=torch.int64, device=source_cloud.device
    )
    previous_fitness = torch.zeros(
        batch_size, dtype=source_cloud.dtype, device=source_cloud.device
    )
    previous_rmse = torch.zeros_like(previous_fitness)
    minimum = 3 if method == "point_to_point" else 6

    for iteration in range(max_iterations):
        evaluation = _evaluate(
            source_cloud,
            target_cloud,
            index,
            transforms,
            max_distance,
            source_ids,
        )
        active = active & (evaluation.counts >= minimum)
        if iteration > 0:
            stable = (
                (evaluation.fitness - previous_fitness).abs() < relative_fitness
            ) & ((evaluation.rmse - previous_rmse).abs() < relative_rmse)
            newly_converged = active & stable
            converged = converged | newly_converged
            active = active & ~newly_converged

        if method == "point_to_point":
            delta, solvable = _point_to_point_delta(
                evaluation,
                active,
                source_ids,
                batch_size,
                robust_kernel,
                robust_delta,
            )
        else:
            assert normals is not None
            delta, solvable = _point_to_plane_delta(
                evaluation,
                normals,
                active,
                source_ids,
                batch_size,
                robust_kernel,
                robust_delta,
            )
        update = active & solvable
        candidate = delta @ transforms
        transforms = torch.where(update[:, None, None], candidate, transforms)
        iterations = iterations + update.to(torch.int64)
        active = update
        previous_fitness = evaluation.fitness
        previous_rmse = evaluation.rmse
        # The scalar check synchronizes CUDA once per iteration, but avoids
        # continuing expensive searches up to max_iterations after every batch
        # entry has converged or failed.
        if not bool(active.any()):
            break

    final = _evaluate(
        source_cloud,
        target_cloud,
        index,
        transforms,
        max_distance,
        source_ids,
    )
    return ICPResult(
        transforms=transforms,
        converged=converged,
        iterations=iterations,
        fitness=final.fitness,
        inlier_rmse=final.rmse,
    )


@torch.no_grad()
def evaluate_registration(
    source: torch.Tensor | PointCloud,
    target: torch.Tensor | PointCloud,
    max_distance: float,
    transforms: torch.Tensor | None = None,
    *,
    index: NeighborIndex | None = None,
) -> RegistrationMetrics:
    """Evaluate source-to-target transforms without performing ICP updates."""
    source_cloud, target_cloud = _prepare_inputs(source, target, max_distance)
    matrices = _initial_transforms(source_cloud, transforms)
    if index is None:
        index = NeighborIndex(target_cloud)
    source_ids = batch_ids(source_cloud.offsets, source_cloud.points.shape[0])
    evaluation = _evaluate(
        source_cloud,
        target_cloud,
        index,
        matrices,
        max_distance,
        source_ids,
    )
    return RegistrationMetrics(
        transforms=matrices,
        fitness=evaluation.fitness,
        inlier_rmse=evaluation.rmse,
    )


__all__ = ["ICPResult", "RegistrationMetrics", "evaluate_registration", "icp"]
