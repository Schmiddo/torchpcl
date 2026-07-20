"""Batched point-cloud distances and reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._segments import segment_mean, segment_sum
from .cloud import PointCloud, PointCloudLike, as_point_cloud
from .neighbors import NeighborIndex
from .validation import check_cloud_pair


@dataclass(frozen=True, eq=False)
class FScoreResult:
    precision: torch.Tensor
    recall: torch.Tensor
    f1_score: torch.Tensor


@dataclass(frozen=True, eq=False)
class PointCloudMetrics:
    accuracy: torch.Tensor
    completion: torch.Tensor
    chamfer_distance: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1_score: torch.Tensor


def _validate_pair(
    source: PointCloud,
    target: PointCloud,
) -> None:
    check_cloud_pair(source, target, "source", "target", non_empty=True)


def _safe_sqrt(values: torch.Tensor) -> torch.Tensor:
    positive = values > 0
    return torch.where(positive, values, torch.ones_like(values)).sqrt() * positive


def _batch_reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _directed_nearest_distance(
    source: PointCloud,
    target: PointCloud,
    *,
    squared: bool = False,
    index: NeighborIndex | None = None,
) -> torch.Tensor:
    """Return one nearest-target distance for every packed source point."""
    if index is None:
        index = NeighborIndex(target)
    distances2 = index.knn(source, 1).distances2[:, 0]
    return distances2 if squared else _safe_sqrt(distances2)


def chamfer_distance(
    source: PointCloudLike,
    target: PointCloudLike,
    *,
    squared: bool = True,
    bidirectional: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute mean nearest-neighbor distances over point-cloud batches.

    Each source point is matched to its nearest target point and the distances
    are averaged per batch entry. When ``bidirectional`` is true, the reverse
    mean distance is included and the two directions are averaged. Distances
    are squared by default; ``reduction`` controls whether the per-batch values
    are returned unchanged, averaged, or summed.
    """
    source_cloud = as_point_cloud(source, "source")
    target_cloud = as_point_cloud(target, "target")
    _validate_pair(source_cloud, target_cloud)

    values = _directed_nearest_distance(
        source_cloud, target_cloud, squared=squared
    )
    result = segment_mean(values, source_cloud.offsets)

    if bidirectional:
        values = _directed_nearest_distance(
            target_cloud, source_cloud, squared=squared
        )
        result = result + segment_mean(values, target_cloud.offsets)
        result = result / 2

    return _batch_reduce(result, reduction)


def fscore(
    prediction: PointCloudLike,
    reference: PointCloudLike,
    threshold: float | torch.Tensor,
) -> FScoreResult:
    """Return per-batch precision, recall, and F-score at one threshold."""
    prediction_cloud = as_point_cloud(prediction, "prediction")
    reference_cloud = as_point_cloud(reference, "reference")
    _validate_pair(prediction_cloud, reference_cloud)
    prediction_distances = _directed_nearest_distance(
        prediction_cloud, reference_cloud
    )
    reference_distances = _directed_nearest_distance(
        reference_cloud, prediction_cloud
    )
    return _fscore_from_distances(
        prediction_distances,
        prediction_cloud,
        reference_distances,
        reference_cloud,
        threshold,
    )


def _fscore_from_distances(
    prediction_distances: torch.Tensor,
    prediction_cloud: PointCloud,
    reference_distances: torch.Tensor,
    reference_cloud: PointCloud,
    threshold: float | torch.Tensor,
) -> FScoreResult:
    threshold_value = torch.as_tensor(
        threshold, dtype=prediction_cloud.dtype, device=prediction_cloud.device
    )
    if threshold_value.ndim != 0:
        raise ValueError("threshold must be a scalar")
    if bool(threshold_value <= 0):
        raise ValueError("threshold must be positive")

    precision = segment_mean(
        (prediction_distances <= threshold_value).to(prediction_cloud.dtype),
        prediction_cloud.offsets,
    )
    recall = segment_mean(
        (reference_distances <= threshold_value).to(reference_cloud.dtype),
        reference_cloud.offsets,
    )
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0,
        2 * precision * recall / denominator.clamp_min(
            torch.finfo(denominator.dtype).tiny
        ),
        torch.zeros_like(denominator),
    )
    return FScoreResult(precision=precision, recall=recall, f1_score=f1)


def point_cloud_metrics(
    prediction: PointCloudLike,
    reference: PointCloudLike,
    threshold: float | torch.Tensor,
) -> PointCloudMetrics:
    """Compute unsquared accuracy, completion, Chamfer, and F-score metrics."""
    prediction_cloud = as_point_cloud(prediction, "prediction")
    reference_cloud = as_point_cloud(reference, "reference")
    _validate_pair(prediction_cloud, reference_cloud)
    prediction_distances = _directed_nearest_distance(
        prediction_cloud, reference_cloud
    )
    reference_distances = _directed_nearest_distance(
        reference_cloud, prediction_cloud
    )
    accuracy = segment_mean(prediction_distances, prediction_cloud.offsets)
    completion = segment_mean(reference_distances, reference_cloud.offsets)
    scores = _fscore_from_distances(
        prediction_distances,
        prediction_cloud,
        reference_distances,
        reference_cloud,
        threshold,
    )
    chamfer = (accuracy + completion) / 2
    return PointCloudMetrics(
        accuracy=accuracy,
        completion=completion,
        chamfer_distance=chamfer,
        precision=scores.precision,
        recall=scores.recall,
        f1_score=scores.f1_score,
    )


__all__ = [
    "FScoreResult",
    "PointCloudMetrics",
    "chamfer_distance",
    "fscore",
    "point_cloud_metrics",
]
