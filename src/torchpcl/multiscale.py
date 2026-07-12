"""Point-cloud pyramids and coarse-to-fine ICP orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .cloud import PointCloud
from .normals import estimate_normals
from .registration import icp
from .types import ICPResult
from .voxel import voxelize


@dataclass(frozen=True)
class ICPScale:
    """Configuration for one coarse-to-fine ICP level."""

    voxel_size: float
    max_distance: float
    iterations: int

    def __post_init__(self) -> None:
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive")
        if not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")


@dataclass(frozen=True, eq=False)
class PointCloudPyramid:
    """Reusable point clouds sampled at an ordered sequence of voxel sizes."""

    levels: tuple[PointCloud, ...]
    voxel_sizes: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.levels or len(self.levels) != len(self.voxel_sizes):
            raise ValueError("levels and voxel_sizes must have the same nonzero length")
        first = self.levels[0]
        for level, voxel_size in zip(self.levels, self.voxel_sizes):
            if not isinstance(level, PointCloud):
                raise TypeError("every pyramid level must be a PointCloud")
            if voxel_size <= 0:
                raise ValueError("voxel_sizes must be positive")
            if (
                level.batch_size != first.batch_size
                or level.device != first.device
                or level.dtype != first.dtype
            ):
                raise ValueError(
                    "all pyramid levels must have the same batch size, device, and dtype"
                )

    @property
    def batch_size(self) -> int:
        return self.levels[0].batch_size

    @property
    def device(self) -> torch.device:
        return self.levels[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.levels[0].dtype


def _as_cloud(value: torch.Tensor | PointCloud) -> PointCloud:
    if isinstance(value, PointCloud):
        return value
    if isinstance(value, torch.Tensor):
        return PointCloud.from_points(value)
    raise TypeError("cloud must be a torch.Tensor or PointCloud")


def _normalized_reduced_normals(
    normals: torch.Tensor,
    partition,
) -> torch.Tensor:
    reduced = partition.reduce(normals, reduction="mean")
    norm = reduced.norm(dim=1, keepdim=True)
    valid = norm[:, 0] > torch.finfo(reduced.dtype).tiny
    return torch.where(
        valid[:, None], reduced / norm.clamp_min(torch.finfo(reduced.dtype).tiny), 0
    )


def build_pyramid(
    cloud: torch.Tensor | PointCloud,
    voxel_sizes: Sequence[float],
    *,
    normal_mode: str = "none",
    normal_k: int = 30,
    normal_radius_factor: float | None = 2.5,
) -> PointCloudPyramid:
    """Voxelize a cloud at each requested resolution.

    ``normal_mode="reduce"`` averages attached input normals per voxel and
    renormalizes them. ``normal_mode="estimate"`` estimates normals independently
    at every level. ``"none"`` omits normals.
    """
    packed = _as_cloud(cloud)
    sizes = tuple(float(size) for size in voxel_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("voxel_sizes must contain positive values")
    if normal_mode not in {"none", "reduce", "estimate"}:
        raise ValueError("normal_mode must be 'none', 'reduce', or 'estimate'")
    if normal_mode == "reduce" and packed.normals is None:
        raise ValueError("normal_mode='reduce' requires attached cloud normals")
    if normal_radius_factor is not None and normal_radius_factor <= 0:
        raise ValueError("normal_radius_factor must be positive or None")

    levels = []
    for voxel_size in sizes:
        partition = voxelize(packed, voxel_size)
        level = partition.cloud
        level_normals = None
        if normal_mode == "reduce":
            assert packed.normals is not None
            level_normals = _normalized_reduced_normals(packed.normals, partition)
        elif normal_mode == "estimate":
            radius = (
                None
                if normal_radius_factor is None
                else voxel_size * normal_radius_factor
            )
            level_normals = estimate_normals(
                level, radius=radius, k=normal_k
            ).normals
        if level_normals is not None:
            level = PointCloud._from_validated(
                level.points, level.offsets, normals=level_normals
            )
        levels.append(level)
    return PointCloudPyramid(tuple(levels), sizes)


def _resolve_pyramid(
    value: torch.Tensor | PointCloud | PointCloudPyramid,
    scales: tuple[ICPScale, ...],
    *,
    normal_mode: str,
    normal_k: int,
    normal_radius_factor: float | None,
) -> PointCloudPyramid:
    sizes = tuple(scale.voxel_size for scale in scales)
    if isinstance(value, PointCloudPyramid):
        if value.voxel_sizes != sizes:
            raise ValueError("pyramid voxel_sizes must match scales")
        if normal_mode == "estimate" and any(
            level.normals is None for level in value.levels
        ):
            levels = []
            for level, scale in zip(value.levels, scales):
                radius = (
                    None
                    if normal_radius_factor is None
                    else scale.voxel_size * normal_radius_factor
                )
                estimated = estimate_normals(
                    level, radius=radius, k=normal_k
                ).normals
                levels.append(
                    PointCloud._from_validated(
                        level.points, level.offsets, normals=estimated
                    )
                )
            return PointCloudPyramid(tuple(levels), sizes)
        return value
    return build_pyramid(
        value,
        sizes,
        normal_mode=normal_mode,
        normal_k=normal_k,
        normal_radius_factor=normal_radius_factor,
    )


@torch.no_grad()
def multiscale_icp(
    source: torch.Tensor | PointCloud | PointCloudPyramid,
    target: torch.Tensor | PointCloud | PointCloudPyramid,
    scales: Sequence[ICPScale],
    *,
    init: torch.Tensor | None = None,
    method: str = "point_to_point",
    target_normals: torch.Tensor | None = None,
    normal_k: int = 30,
    normal_radius_factor: float | None = 2.5,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
    robust_kernel: str | None = None,
    robust_delta: float = 1.0,
) -> ICPResult:
    """Run coarse-to-fine ICP and return metrics from the finest level.

    Iteration counts are accumulated across levels. ``converged`` and the error
    metrics describe the final level. Precomputed pyramids can be passed
    directly as ``source`` and ``target``.
    """
    scale_tuple = tuple(scales)
    if not scale_tuple or any(not isinstance(scale, ICPScale) for scale in scale_tuple):
        raise ValueError("scales must contain at least one ICPScale")
    if method not in {"point_to_point", "point_to_plane"}:
        raise ValueError("method must be 'point_to_point' or 'point_to_plane'")
    if target_normals is not None:
        if isinstance(target, PointCloudPyramid):
            raise ValueError("target_normals cannot be combined with a target pyramid")
        target_cloud = _as_cloud(target)
        if target_cloud.normals is not None:
            raise ValueError("target normals were specified twice")
        target = PointCloud(
            target_cloud.points,
            target_cloud.offsets,
            normals=target_normals,
            features=target_cloud.features,
        )

    source_pyramid = _resolve_pyramid(
        source,
        scale_tuple,
        normal_mode="none",
        normal_k=normal_k,
        normal_radius_factor=normal_radius_factor,
    )
    if method == "point_to_plane":
        if isinstance(target, PointCloudPyramid):
            normal_mode = "estimate"
        else:
            target_cloud = _as_cloud(target)
            normal_mode = "reduce" if target_cloud.normals is not None else "estimate"
    else:
        normal_mode = "none"
    target_pyramid = _resolve_pyramid(
        target,
        scale_tuple,
        normal_mode=normal_mode,
        normal_k=normal_k,
        normal_radius_factor=normal_radius_factor,
    )
    if (
        source_pyramid.batch_size != target_pyramid.batch_size
        or source_pyramid.device != target_pyramid.device
        or source_pyramid.dtype != target_pyramid.dtype
    ):
        raise ValueError(
            "source and target pyramids must match in batch size, device, and dtype"
        )

    transforms = init
    total_iterations = torch.zeros(
        source_pyramid.batch_size,
        dtype=torch.int64,
        device=source_pyramid.device,
    )
    result = None
    for source_level, target_level, scale in zip(
        source_pyramid.levels, target_pyramid.levels, scale_tuple
    ):
        result = icp(
            source_level,
            target_level,
            scale.max_distance,
            init=transforms,
            method=method,
            target_normals=target_level.normals,
            max_iterations=scale.iterations,
            relative_fitness=relative_fitness,
            relative_rmse=relative_rmse,
            robust_kernel=robust_kernel,
            robust_delta=robust_delta,
        )
        transforms = result.transforms
        total_iterations = total_iterations + result.iterations

    assert result is not None
    return ICPResult(
        transforms=result.transforms,
        converged=result.converged,
        iterations=total_iterations,
        fitness=result.fitness,
        inlier_rmse=result.inlier_rmse,
    )


__all__ = ["ICPScale", "PointCloudPyramid", "build_pyramid", "multiscale_icp"]
