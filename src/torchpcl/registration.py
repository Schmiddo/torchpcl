"""Batched single- and multi-level iterative closest point registration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ._segments import segment_sum
from .alignment import _procrustes_packed
from .cloud import PointCloud, PointCloudLike, _normalize_cloud, batch_ids
from .neighbors import NeighborIndex
from .transforms import transform
from .validation import check_cloud_pair
from .voxel import voxelize


@dataclass(frozen=True, kw_only=True)
class ICPLevel:
    """Configuration for one ICP level."""

    max_correspondence_distance: float
    max_iterations: int = 30
    voxel_size: float | None = None

    def __post_init__(self) -> None:
        if not self.max_correspondence_distance > 0:
            raise ValueError("max_correspondence_distance must be positive")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        if self.voxel_size is not None and not self.voxel_size > 0:
            raise ValueError("voxel_size must be positive or None")


@dataclass(frozen=True)
class PointToPoint:
    """Point-to-point ICP objective."""


@dataclass(frozen=True)
class PointToPlane:
    """Point-to-plane ICP objective."""


@dataclass(frozen=True, kw_only=True)
class HuberLoss:
    """Huber residual weighting."""

    delta: float = 1.0

    def __post_init__(self) -> None:
        if not self.delta > 0:
            raise ValueError("delta must be positive")


@dataclass(frozen=True, kw_only=True)
class L1Loss:
    """L1 residual weighting."""


@dataclass(frozen=True, kw_only=True)
class CauchyLoss:
    """Cauchy (Lorentzian) residual weighting."""

    delta: float = 1.0

    def __post_init__(self) -> None:
        if not self.delta > 0:
            raise ValueError("delta must be positive")


@dataclass(frozen=True, kw_only=True)
class GMLoss:
    """Geman-McClure residual weighting."""

    delta: float = 1.0

    def __post_init__(self) -> None:
        if not self.delta > 0:
            raise ValueError("delta must be positive")


@dataclass(frozen=True, kw_only=True)
class TukeyLoss:
    """Tukey biweight residual weighting."""

    delta: float = 1.0

    def __post_init__(self) -> None:
        if not self.delta > 0:
            raise ValueError("delta must be positive")


RobustLoss = HuberLoss | L1Loss | CauchyLoss | GMLoss | TukeyLoss
_ROBUST_LOSS_TYPES = (HuberLoss, L1Loss, CauchyLoss, GMLoss, TukeyLoss)


@dataclass(frozen=True, kw_only=True)
class ConvergenceCriteria:
    """Absolute fitness and RMSE change tolerances."""

    fitness_tolerance: float = 1e-6
    rmse_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not self.fitness_tolerance >= 0:
            raise ValueError("fitness_tolerance must be nonnegative")
        if not self.rmse_tolerance >= 0:
            raise ValueError("rmse_tolerance must be nonnegative")


@dataclass(frozen=True, kw_only=True)
class ICPOptions:
    """Options shared by every level of an ICP call."""

    objective: PointToPoint | PointToPlane = field(default_factory=PointToPoint)
    convergence: ConvergenceCriteria = field(default_factory=ConvergenceCriteria)
    robust_loss: RobustLoss | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.objective, (PointToPoint, PointToPlane)):
            raise TypeError("objective must be PointToPoint or PointToPlane")
        if not isinstance(self.convergence, ConvergenceCriteria):
            raise TypeError("convergence must be ConvergenceCriteria")
        if self.robust_loss is not None and not isinstance(
            self.robust_loss, _ROBUST_LOSS_TYPES
        ):
            raise TypeError(
                "robust_loss must be HuberLoss, L1Loss, CauchyLoss, GMLoss, "
                "TukeyLoss, or None"
            )


@dataclass(frozen=True, eq=False)
class RegistrationMetrics:
    """Metrics for a batch of source-to-target transforms."""

    transforms: torch.Tensor
    fitness: torch.Tensor
    inlier_rmse: torch.Tensor


@dataclass(frozen=True, eq=False)
class ICPLevelResult:
    """Batched ICP output for one level."""

    level: ICPLevel
    transforms: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor
    fitness: torch.Tensor
    inlier_rmse: torch.Tensor


@dataclass(frozen=True, eq=False)
class ICPResult:
    """Batched output from a complete ICP level sequence."""

    transforms: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor
    fitness: torch.Tensor
    inlier_rmse: torch.Tensor
    level_results: tuple[ICPLevelResult, ...]


@dataclass(frozen=True, eq=False)
class _Evaluation:
    current: torch.Tensor
    target: torch.Tensor
    indices: torch.Tensor
    valid: torch.Tensor
    counts: torch.Tensor
    fitness: torch.Tensor
    rmse: torch.Tensor


@dataclass(frozen=True, eq=False)
class _PreparedLevel:
    config: ICPLevel
    source: PointCloud
    target: PointCloud
    target_normals: torch.Tensor | None
    index: NeighborIndex


def _prepare_cloud_pair(
    source: PointCloudLike,
    target: PointCloudLike,
) -> tuple[PointCloud, PointCloud]:
    source_cloud = _normalize_cloud(source, "source").cloud
    target_cloud = _normalize_cloud(target, "target").cloud
    check_cloud_pair(source_cloud, target_cloud, "source", "target", non_empty=True)
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


def _reduced_normals(
    normals: torch.Tensor,
    partition,
) -> torch.Tensor:
    reduced = partition.reduce(normals, reduction="mean")
    norm = reduced.norm(dim=1, keepdim=True)
    tiny = torch.finfo(reduced.dtype).tiny
    valid = norm[:, 0] > tiny
    return torch.where(valid[:, None], reduced / norm.clamp_min(tiny), 0)


def _materialize_level(
    source: PointCloud,
    target: PointCloud,
    config: ICPLevel,
    objective: PointToPoint | PointToPlane,
) -> _PreparedLevel:
    if config.voxel_size is None:
        source_level = source
        target_level = target
    else:
        source_level = voxelize(source, config.voxel_size).cloud
        target_partition = voxelize(target, config.voxel_size)
        target_level = target_partition.cloud
        if isinstance(objective, PointToPlane):
            assert target.normals is not None
            normals = _reduced_normals(target.normals, target_partition)
            target_level = PointCloud._from_validated(
                target_level.points,
                target_level.offsets,
                normals=normals,
            )

    target_normals = (
        target_level.normals if isinstance(objective, PointToPlane) else None
    )
    return _PreparedLevel(
        config=config,
        source=source_level,
        target=target_level,
        target_normals=target_normals,
        index=NeighborIndex(target_level),
    )


def _evaluate(
    level: _PreparedLevel,
    transforms: torch.Tensor,
) -> _Evaluation:
    current_cloud = transform(level.source, transforms)
    assert isinstance(current_cloud, PointCloud)
    neighbors = level.index.hybrid(
        current_cloud,
        level.config.max_correspondence_distance,
        1,
    )
    indices = neighbors.indices[:, 0]
    valid = neighbors.valid[:, 0]
    target_points = level.target.points[indices.clamp(min=0)]
    distances2 = neighbors.distances2[:, 0].masked_fill(~valid, 0)
    counts = segment_sum(valid.to(level.source.dtype), level.source.offsets)
    squared_error = segment_sum(distances2, level.source.offsets)
    fitness = counts / level.source.lengths.to(level.source.dtype)
    rmse = torch.where(
        counts > 0,
        (squared_error / counts.clamp(min=1)).sqrt(),
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


def _kernel_weight(
    residual_magnitude: torch.Tensor,
    robust_loss: RobustLoss,
) -> torch.Tensor:
    magnitude = residual_magnitude.clamp_min(
        torch.finfo(residual_magnitude.dtype).tiny
    )
    if isinstance(robust_loss, HuberLoss):
        return torch.where(
            residual_magnitude <= robust_loss.delta,
            torch.ones_like(magnitude),
            robust_loss.delta / magnitude,
        )
    if isinstance(robust_loss, L1Loss):
        return 1.0 / magnitude
    if isinstance(robust_loss, CauchyLoss):
        return 1.0 / (1.0 + (residual_magnitude / robust_loss.delta) ** 2)
    if isinstance(robust_loss, GMLoss):
        # Matches Open3D's RobustKernel GMLoss::Weight exactly, including its
        # k^2 / (k^2 + r^2)^2 normalization (weight(0) = 1/delta^2, not 1).
        # A per-kernel constant scale factor doesn't change the weighted
        # least-squares solution, so this is intentional, not a bug.
        delta2 = robust_loss.delta**2
        denom = delta2 + residual_magnitude**2
        return delta2 / (denom * denom)
    assert isinstance(robust_loss, TukeyLoss)
    ratio2 = (residual_magnitude / robust_loss.delta) ** 2
    return torch.where(
        residual_magnitude <= robust_loss.delta,
        (1.0 - ratio2) ** 2,
        torch.zeros_like(ratio2),
    )


def _robust_weights(
    residual_magnitude: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    ids: torch.Tensor,
    robust_loss: RobustLoss | None,
) -> torch.Tensor:
    weights = (valid & active[ids]).to(residual_magnitude.dtype)
    if robust_loss is not None:
        weights = weights * _kernel_weight(residual_magnitude, robust_loss)
    return weights


def _point_to_point_delta(
    evaluation: _Evaluation,
    active: torch.Tensor,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    robust_loss: RobustLoss | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = (evaluation.current - evaluation.target).norm(dim=1)
    weights = _robust_weights(
        residual,
        evaluation.valid,
        active,
        ids,
        robust_loss,
    )
    alignment, _ = _procrustes_packed(
        evaluation.current,
        evaluation.target,
        weights,
        ids,
        offsets,
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
    offsets: torch.Tensor,
    robust_loss: RobustLoss | None,
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
        robust_loss,
    )
    jacobian = torch.cat(
        [torch.linalg.cross(evaluation.current, matched_normals), matched_normals],
        dim=1,
    )
    jtj = segment_sum(
        jacobian[:, :, None] * jacobian[:, None, :] * weights[:, None, None],
        offsets,
    )
    jtr = segment_sum(jacobian * residual[:, None] * weights[:, None], offsets)
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


def _run_level(
    level: _PreparedLevel,
    transforms: torch.Tensor,
    options: ICPOptions,
) -> ICPLevelResult:
    source = level.source
    source_ids = batch_ids(source.offsets, source.points.shape[0])
    active = torch.ones(
        source.batch_size,
        dtype=torch.bool,
        device=source.device,
    )
    converged = torch.zeros_like(active)
    iterations = torch.zeros(
        source.batch_size,
        dtype=torch.int64,
        device=source.device,
    )
    previous_fitness = torch.zeros(
        source.batch_size,
        dtype=source.dtype,
        device=source.device,
    )
    previous_rmse = torch.zeros_like(previous_fitness)
    minimum = 3 if isinstance(options.objective, PointToPoint) else 6

    for iteration in range(level.config.max_iterations):
        evaluation = _evaluate(level, transforms)
        active = active & (evaluation.counts >= minimum)
        if iteration > 0:
            stable = (
                (evaluation.fitness - previous_fitness).abs()
                < options.convergence.fitness_tolerance
            ) & (
                (evaluation.rmse - previous_rmse).abs()
                < options.convergence.rmse_tolerance
            )
            newly_converged = active & stable
            converged = converged | newly_converged
            active = active & ~newly_converged

        if isinstance(options.objective, PointToPoint):
            delta, solvable = _point_to_point_delta(
                evaluation,
                active,
                source_ids,
                source.offsets,
                options.robust_loss,
            )
        else:
            assert level.target_normals is not None
            delta, solvable = _point_to_plane_delta(
                evaluation,
                level.target_normals,
                active,
                source_ids,
                source.offsets,
                options.robust_loss,
            )
        update = active & solvable
        candidate = delta @ transforms
        transforms = torch.where(update[:, None, None], candidate, transforms)
        iterations = iterations + update.to(torch.int64)
        active = update
        previous_fitness = evaluation.fitness
        previous_rmse = evaluation.rmse
        if not bool(active.any()):
            break

    final = _evaluate(level, transforms)
    return ICPLevelResult(
        level=level.config,
        transforms=transforms,
        converged=converged,
        iterations=iterations,
        fitness=final.fitness,
        inlier_rmse=final.rmse,
    )


@torch.no_grad()
def icp(
    source: PointCloudLike,
    target: PointCloudLike,
    levels: Sequence[ICPLevel],
    *,
    init: torch.Tensor | None = None,
    options: ICPOptions | None = None,
) -> ICPResult:
    """Register corresponding clouds through one or more ICP levels."""
    level_configs = tuple(levels)
    if not level_configs:
        raise ValueError("levels must contain at least one ICPLevel")
    if any(not isinstance(level, ICPLevel) for level in level_configs):
        raise TypeError("levels must contain only ICPLevel instances")
    if options is None:
        options = ICPOptions()
    elif not isinstance(options, ICPOptions):
        raise TypeError("options must be ICPOptions or None")

    source_cloud, target_cloud = _prepare_cloud_pair(source, target)
    if isinstance(options.objective, PointToPlane) and target_cloud.normals is None:
        raise ValueError("PointToPlane requires target normals")

    transforms = _initial_transforms(source_cloud, init)
    total_iterations = torch.zeros(
        source_cloud.batch_size,
        dtype=torch.int64,
        device=source_cloud.device,
    )
    level_results = []
    for config in level_configs:
        prepared = _materialize_level(
            source_cloud,
            target_cloud,
            config,
            options.objective,
        )
        level_result = _run_level(prepared, transforms, options)
        level_results.append(level_result)
        transforms = level_result.transforms
        total_iterations = total_iterations + level_result.iterations

    final = level_results[-1]
    return ICPResult(
        transforms=final.transforms,
        converged=final.converged,
        iterations=total_iterations,
        fitness=final.fitness,
        inlier_rmse=final.inlier_rmse,
        level_results=tuple(level_results),
    )


@torch.no_grad()
def evaluate_registration(
    source: PointCloudLike,
    target: PointCloudLike,
    max_correspondence_distance: float,
    transforms: torch.Tensor | None = None,
) -> RegistrationMetrics:
    """Evaluate source-to-target transforms without performing ICP updates."""
    if not max_correspondence_distance > 0:
        raise ValueError("max_correspondence_distance must be positive")
    source_cloud, target_cloud = _prepare_cloud_pair(source, target)
    matrices = _initial_transforms(source_cloud, transforms)
    level = _PreparedLevel(
        config=ICPLevel(
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=0,
        ),
        source=source_cloud,
        target=target_cloud,
        target_normals=None,
        index=NeighborIndex(target_cloud),
    )
    evaluation = _evaluate(level, matrices)
    return RegistrationMetrics(
        transforms=matrices,
        fitness=evaluation.fitness,
        inlier_rmse=evaluation.rmse,
    )


__all__ = [
    "CauchyLoss",
    "ConvergenceCriteria",
    "GMLoss",
    "HuberLoss",
    "ICPLevel",
    "ICPLevelResult",
    "ICPOptions",
    "ICPResult",
    "L1Loss",
    "PointToPlane",
    "PointToPoint",
    "RegistrationMetrics",
    "RobustLoss",
    "TukeyLoss",
    "evaluate_registration",
    "icp",
]
