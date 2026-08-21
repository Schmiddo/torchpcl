"""Fast global registration from features or explicit correspondences."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .cloud import PointCloud, PointCloudLike, as_point_cloud
from .feature_matching import CorrespondenceSet, match_features
from .registration import evaluate_registration
from .transforms import pose_to_matrix
from .validation import check_cloud_pair


@dataclass(frozen=True, kw_only=True)
class FGROptions:
    """Small option surface for the fast global registration prototype."""

    max_correspondence_distance: float
    max_iterations: int = 64
    tuple_scale: float | None = 0.95
    max_tuple_count: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_correspondence_distance, (int, float))
            or not math.isfinite(self.max_correspondence_distance)
            or self.max_correspondence_distance <= 0
        ):
            raise ValueError("max_correspondence_distance must be finite and positive")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        if self.tuple_scale is not None and (
            not isinstance(self.tuple_scale, (int, float))
            or not math.isfinite(self.tuple_scale)
            or not 0 < self.tuple_scale < 1
        ):
            raise ValueError("tuple_scale must be in (0, 1) or None")
        if not isinstance(self.max_tuple_count, int) or self.max_tuple_count <= 0:
            raise ValueError("max_tuple_count must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True, eq=False)
class FGRResult:
    """Single-pair FGR output with batch-shaped registration metrics."""

    transforms: torch.Tensor
    success: torch.Tensor
    iterations: torch.Tensor
    correspondence_counts: torch.Tensor
    fitness: torch.Tensor
    inlier_rmse: torch.Tensor


def _validate_pair(
    source: PointCloudLike,
    target: PointCloudLike,
) -> tuple[PointCloud, PointCloud]:
    source_cloud = as_point_cloud(source, "source")
    target_cloud = as_point_cloud(target, "target")
    check_cloud_pair(
        source_cloud, target_cloud, "source", "target", non_empty=True
    )
    if source_cloud.batch_size != 1:
        raise ValueError("fast global registration currently supports batch_size == 1")
    return source_cloud, target_cloud


def _validate_correspondences(
    correspondences: CorrespondenceSet,
    source: PointCloud,
    target: PointCloud,
) -> None:
    if not isinstance(correspondences, CorrespondenceSet):
        raise TypeError("correspondences must be a CorrespondenceSet")
    if correspondences.source_indices.device != source.device:
        raise ValueError("correspondences and point clouds must be on the same device")
    if correspondences.count == 0:
        return
    source_valid = (correspondences.source_indices >= 0) & (
        correspondences.source_indices < source.points.shape[0]
    )
    target_valid = (correspondences.target_indices >= 0) & (
        correspondences.target_indices < target.points.shape[0]
    )
    if not bool((source_valid & target_valid).all()):
        raise ValueError("correspondence indices must address source and target rows")


def _tuple_filter(
    source: PointCloud,
    target: PointCloud,
    correspondences: CorrespondenceSet,
    options: FGROptions,
) -> CorrespondenceSet:
    if options.tuple_scale is None or correspondences.count < 3:
        return correspondences

    trial_count = min(correspondences.count * 100, 100_000)
    generator = torch.Generator(device="cpu").manual_seed(options.seed)
    samples = torch.randint(
        correspondences.count,
        (trial_count, 3),
        generator=generator,
        device="cpu",
    ).to(source.device)
    source_rows = correspondences.source_indices[samples]
    target_rows = correspondences.target_indices[samples]
    source_triples = source.points[source_rows]
    target_triples = target.points[target_rows]

    source_edges = torch.stack(
        (
            source_triples[:, 0] - source_triples[:, 1],
            source_triples[:, 1] - source_triples[:, 2],
            source_triples[:, 2] - source_triples[:, 0],
        ),
        dim=1,
    ).norm(dim=2)
    target_edges = torch.stack(
        (
            target_triples[:, 0] - target_triples[:, 1],
            target_triples[:, 1] - target_triples[:, 2],
            target_triples[:, 2] - target_triples[:, 0],
        ),
        dim=1,
    ).norm(dim=2)
    scale = float(options.tuple_scale)
    compatible = (
        (source_edges * scale < target_edges)
        & (target_edges < source_edges / scale)
    ).all(dim=1)
    accepted = samples[compatible][: options.max_tuple_count].reshape(-1)
    source_indices = correspondences.source_indices[accepted]
    target_indices = correspondences.target_indices[accepted]
    distances2 = (
        None
        if correspondences.distances2 is None
        else correspondences.distances2[accepted]
    )
    return CorrespondenceSet(source_indices, target_indices, distances2)


def _normalized_transform_to_original(
    transform: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    rotation = transform[:3, :3]
    normalized_translation = transform[:3, 3]
    translation = (
        target_mean
        - rotation @ source_mean
        + scale * normalized_translation
    )
    result = transform.clone()
    result[:3, 3] = translation
    return result


def _failed_result(
    source: PointCloud,
    target: PointCloud,
    options: FGROptions,
    correspondence_count: int,
) -> FGRResult:
    transform = torch.eye(4, dtype=source.dtype, device=source.device).unsqueeze(0)
    metrics = evaluate_registration(
        source, target, options.max_correspondence_distance, transform
    )
    return FGRResult(
        transforms=transform,
        success=torch.zeros(1, dtype=torch.bool, device=source.device),
        iterations=torch.zeros(1, dtype=torch.int64, device=source.device),
        correspondence_counts=torch.tensor(
            [correspondence_count], dtype=torch.int64, device=source.device
        ),
        fitness=metrics.fitness,
        inlier_rmse=metrics.inlier_rmse,
    )


@torch.no_grad()
def fast_global_registration_from_correspondences(
    source: PointCloudLike,
    target: PointCloudLike,
    correspondences: CorrespondenceSet,
    *,
    options: FGROptions,
) -> FGRResult:
    """Estimate a coarse source-to-target transform for one cloud pair."""
    if not isinstance(options, FGROptions):
        raise TypeError("options must be FGROptions")
    source_cloud, target_cloud = _validate_pair(source, target)
    _validate_correspondences(correspondences, source_cloud, target_cloud)
    filtered = _tuple_filter(source_cloud, target_cloud, correspondences, options)
    if filtered.count < 10:
        return _failed_result(
            source_cloud, target_cloud, options, correspondences.count
        )

    source_mean = source_cloud.points.mean(dim=0)
    target_mean = target_cloud.points.mean(dim=0)
    centered_source = source_cloud.points - source_mean
    centered_target = target_cloud.points - target_mean
    scale = torch.maximum(
        centered_source.norm(dim=1).max(), centered_target.norm(dim=1).max()
    )
    if not bool(torch.isfinite(scale)) or not bool(scale > 0):
        return _failed_result(
            source_cloud, target_cloud, options, correspondences.count
        )

    matched_source = centered_source[filtered.source_indices] / scale
    matched_target = centered_target[filtered.target_indices] / scale
    transform = torch.eye(4, dtype=source_cloud.dtype, device=source_cloud.device)
    mu = source_cloud.points.new_tensor(1.0)
    maximum_distance = source_cloud.points.new_tensor(
        options.max_correspondence_distance
    )
    minimum_mu = (maximum_distance / scale).square()
    completed = 0

    for iteration in range(options.max_iterations):
        current = matched_source @ transform[:3, :3].T + transform[:3, 3]
        residual = current - matched_target
        residual2 = residual.square().sum(dim=1)
        weights = (mu / (residual2 + mu)).square()

        x, y, z = current.unbind(dim=1)
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        jacobian = torch.stack(
            (
                torch.stack((zeros, z, -y, ones, zeros, zeros), dim=1),
                torch.stack((-z, zeros, x, zeros, ones, zeros), dim=1),
                torch.stack((y, -x, zeros, zeros, zeros, ones), dim=1),
            ),
            dim=1,
        )
        weighted_jacobian = jacobian * weights[:, None, None]
        jtj = (jacobian.transpose(1, 2) @ weighted_jacobian).sum(dim=0)
        jtr = (
            jacobian.transpose(1, 2)
            @ (residual * weights[:, None])[:, :, None]
        ).sum(dim=0)[:, 0]
        pose, info = torch.linalg.solve_ex(jtj, -jtr)
        if bool(info != 0) or not bool(torch.isfinite(pose).all()):
            return _failed_result(
                source_cloud, target_cloud, options, correspondences.count
            )

        delta = pose_to_matrix(pose).to(source_cloud.dtype)
        candidate = delta @ transform
        if not bool(torch.isfinite(candidate).all()):
            return _failed_result(
                source_cloud, target_cloud, options, correspondences.count
            )
        transform = candidate
        completed = iteration + 1
        if iteration % 4 == 0 and bool(mu > minimum_mu):
            mu = torch.maximum(mu / 1.4, minimum_mu)

    original_transform = _normalized_transform_to_original(
        transform, source_mean, target_mean, scale
    ).unsqueeze(0)
    metrics = evaluate_registration(
        source_cloud,
        target_cloud,
        options.max_correspondence_distance,
        original_transform,
    )
    return FGRResult(
        transforms=original_transform,
        success=torch.ones(1, dtype=torch.bool, device=source_cloud.device),
        iterations=torch.tensor(
            [completed], dtype=torch.int64, device=source_cloud.device
        ),
        correspondence_counts=torch.tensor(
            [correspondences.count], dtype=torch.int64, device=source_cloud.device
        ),
        fitness=metrics.fitness,
        inlier_rmse=metrics.inlier_rmse,
    )


@torch.no_grad()
def fast_global_registration(
    source: PointCloud,
    target: PointCloud,
    *,
    options: FGROptions,
) -> FGRResult:
    """Match attached point features and estimate a coarse rigid transform."""
    if not isinstance(source, PointCloud) or not isinstance(target, PointCloud):
        raise TypeError("source and target must be PointCloud objects")
    correspondences = match_features(source, target, mutual=True)
    return fast_global_registration_from_correspondences(
        source, target, correspondences, options=options
    )


__all__ = [
    "FGROptions",
    "FGRResult",
    "fast_global_registration",
    "fast_global_registration_from_correspondences",
]
