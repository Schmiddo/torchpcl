"""Batched point-cloud distances and reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._segments import segment_sum
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


def _point_reduce(
    values: torch.Tensor,
    cloud: PointCloud,
    reduction: str,
) -> torch.Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError("point_reduction must be 'mean' or 'sum'")
    result = segment_sum(values, cloud.offsets)
    if reduction == "mean":
        shape = (cloud.batch_size, *([1] * (values.ndim - 1)))
        result = result / cloud.lengths.to(values.dtype).reshape(shape)
    return result


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
    directional: str = "both",
    combine: str = "mean",
    point_reduction: str = "mean",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute directed or bidirectional Chamfer distance over packed batches."""
    source_cloud = as_point_cloud(source, "source")
    target_cloud = as_point_cloud(target, "target")
    _validate_pair(source_cloud, target_cloud)
    if directional not in {"both", "source_to_target", "target_to_source"}:
        raise ValueError(
            "directional must be 'both', 'source_to_target', or 'target_to_source'"
        )
    if combine not in {"mean", "sum"}:
        raise ValueError("combine must be 'mean' or 'sum'")

    directed_values = []
    if directional in {"both", "source_to_target"}:
        values = _directed_nearest_distance(
            source_cloud, target_cloud, squared=squared
        )
        directed_values.append(_point_reduce(values, source_cloud, point_reduction))
    if directional in {"both", "target_to_source"}:
        values = _directed_nearest_distance(
            target_cloud, source_cloud, squared=squared
        )
        directed_values.append(_point_reduce(values, target_cloud, point_reduction))

    result = directed_values[0]
    if len(directed_values) == 2:
        result = result + directed_values[1]
        if combine == "mean":
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

    precision = _point_reduce(
        (prediction_distances <= threshold_value).to(prediction_cloud.dtype),
        prediction_cloud,
        "mean",
    )
    recall = _point_reduce(
        (reference_distances <= threshold_value).to(reference_cloud.dtype),
        reference_cloud,
        "mean",
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
    accuracy = _point_reduce(prediction_distances, prediction_cloud, "mean")
    completion = _point_reduce(reference_distances, reference_cloud, "mean")
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
