"""Batched point-cloud distances and reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cloud import PointCloud, batch_ids
from .neighbors import NeighborIndex


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


def _as_cloud(value: torch.Tensor | PointCloud, name: str) -> PointCloud:
    if isinstance(value, PointCloud):
        return value
    if isinstance(value, torch.Tensor):
        return PointCloud.from_points(value)
    raise TypeError(f"{name} must be a torch.Tensor or PointCloud")


def _validate_pair(
    source: torch.Tensor | PointCloud,
    target: torch.Tensor | PointCloud,
) -> tuple[PointCloud, PointCloud, bool]:
    unbatched = isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor)
    source_cloud = _as_cloud(source, "source")
    target_cloud = _as_cloud(target, "target")
    if source_cloud.batch_size != target_cloud.batch_size:
        raise ValueError("source and target must have the same batch size")
    if source_cloud.device != target_cloud.device:
        raise ValueError("source and target must be on the same device")
    if source_cloud.dtype != target_cloud.dtype:
        raise ValueError("source and target must have the same dtype")
    has_empty = torch.any(
        (source_cloud.lengths == 0) | (target_cloud.lengths == 0)
    )
    if bool(has_empty):
        raise ValueError("source and target batches must be non-empty")
    return source_cloud, target_cloud, unbatched


def _safe_sqrt(values: torch.Tensor) -> torch.Tensor:
    positive = values > 0
    return torch.where(positive, values, torch.ones_like(values)).sqrt() * positive


def _segment_sum(values: torch.Tensor, cloud: PointCloud) -> torch.Tensor:
    ids = batch_ids(cloud.offsets, cloud.points.shape[0])
    output = values.new_zeros((cloud.batch_size, *values.shape[1:]))
    output.index_add_(0, ids, values)
    return output


def _point_reduce(
    values: torch.Tensor,
    cloud: PointCloud,
    reduction: str,
) -> torch.Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError("point_reduction must be 'mean' or 'sum'")
    result = _segment_sum(values, cloud)
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


def directed_nearest_distance(
    source: torch.Tensor | PointCloud,
    target: torch.Tensor | PointCloud,
    *,
    squared: bool = False,
    index: NeighborIndex | None = None,
) -> torch.Tensor:
    """Return one nearest-target distance for every packed source point."""
    source_cloud, target_cloud, _ = _validate_pair(source, target)
    if index is None:
        index = NeighborIndex(target_cloud)
    elif index.reference.points.data_ptr() != target_cloud.points.data_ptr():
        raise ValueError("index must have been built for target")
    distances2 = index.knn(source_cloud, 1).distances2[:, 0]
    return distances2 if squared else _safe_sqrt(distances2)


def chamfer_distance(
    source: torch.Tensor | PointCloud,
    target: torch.Tensor | PointCloud,
    *,
    squared: bool = True,
    directional: str = "both",
    combine: str = "mean",
    point_reduction: str = "mean",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute directed or bidirectional Chamfer distance over packed batches."""
    source_cloud, target_cloud, _ = _validate_pair(source, target)
    if directional not in {"both", "source_to_target", "target_to_source"}:
        raise ValueError(
            "directional must be 'both', 'source_to_target', or 'target_to_source'"
        )
    if combine not in {"mean", "sum"}:
        raise ValueError("combine must be 'mean' or 'sum'")

    directed_values = []
    if directional in {"both", "source_to_target"}:
        values = directed_nearest_distance(
            source_cloud, target_cloud, squared=squared
        )
        directed_values.append(_point_reduce(values, source_cloud, point_reduction))
    if directional in {"both", "target_to_source"}:
        values = directed_nearest_distance(
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
    prediction: torch.Tensor | PointCloud,
    reference: torch.Tensor | PointCloud,
    threshold: float | torch.Tensor,
) -> FScoreResult:
    """Return per-batch precision, recall, and F-score at one or more thresholds."""
    prediction_cloud, reference_cloud, unbatched = _validate_pair(
        prediction, reference
    )
    prediction_distances = directed_nearest_distance(
        prediction_cloud, reference_cloud
    )
    reference_distances = directed_nearest_distance(
        reference_cloud, prediction_cloud
    )
    return _fscore_from_distances(
        prediction_distances,
        prediction_cloud,
        reference_distances,
        reference_cloud,
        threshold,
        unbatched=unbatched,
    )


def _fscore_from_distances(
    prediction_distances: torch.Tensor,
    prediction_cloud: PointCloud,
    reference_distances: torch.Tensor,
    reference_cloud: PointCloud,
    threshold: float | torch.Tensor,
    *,
    unbatched: bool,
) -> FScoreResult:
    thresholds = torch.as_tensor(
        threshold, dtype=prediction_cloud.dtype, device=prediction_cloud.device
    )
    scalar_threshold = thresholds.ndim == 0
    if thresholds.ndim > 1:
        raise ValueError("threshold must be a scalar or one-dimensional tensor")
    thresholds = thresholds.reshape(-1)
    if bool(torch.any(thresholds <= 0)):
        raise ValueError("threshold must be positive")

    precision = _point_reduce(
        (prediction_distances[:, None] <= thresholds).to(prediction_cloud.dtype),
        prediction_cloud,
        "mean",
    )
    recall = _point_reduce(
        (reference_distances[:, None] <= thresholds).to(reference_cloud.dtype),
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
    if scalar_threshold:
        precision, recall, f1 = precision[:, 0], recall[:, 0], f1[:, 0]
    if unbatched:
        precision, recall, f1 = precision.squeeze(0), recall.squeeze(0), f1.squeeze(0)
    return FScoreResult(precision=precision, recall=recall, f1_score=f1)


def point_cloud_metrics(
    prediction: torch.Tensor | PointCloud,
    reference: torch.Tensor | PointCloud,
    threshold: float | torch.Tensor,
) -> PointCloudMetrics:
    """Compute unsquared accuracy, completion, Chamfer, and F-score metrics."""
    prediction_cloud, reference_cloud, unbatched = _validate_pair(
        prediction, reference
    )
    prediction_distances = directed_nearest_distance(
        prediction_cloud, reference_cloud
    )
    reference_distances = directed_nearest_distance(
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
        unbatched=False,
    )
    chamfer = (accuracy + completion) / 2
    if unbatched:
        accuracy = accuracy.squeeze(0)
        completion = completion.squeeze(0)
        chamfer = chamfer.squeeze(0)
        precision = scores.precision.squeeze(0)
        recall = scores.recall.squeeze(0)
        f1 = scores.f1_score.squeeze(0)
    else:
        precision, recall, f1 = (
            scores.precision,
            scores.recall,
            scores.f1_score,
        )
    return PointCloudMetrics(
        accuracy=accuracy,
        completion=completion,
        chamfer_distance=chamfer,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def chamfer_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    prediction_lengths: torch.Tensor | None = None,
    reference_lengths: torch.Tensor | None = None,
    squared: bool = True,
    single_directional: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compatibility adapter for unbatched or padded dense Chamfer inputs."""
    if prediction.ndim != reference.ndim or prediction.ndim not in {2, 3}:
        raise ValueError(
            "prediction and reference must both have shape (N, 3) or (B, N, 3)"
        )
    if prediction.ndim == 2:
        if prediction_lengths is not None or reference_lengths is not None:
            raise ValueError("lengths are only valid for padded inputs")
        source: torch.Tensor | PointCloud = prediction
        target: torch.Tensor | PointCloud = reference
    else:
        if prediction.shape[0] != reference.shape[0]:
            raise ValueError("prediction and reference must have the same batch size")
        if prediction_lengths is None:
            prediction_lengths = torch.full(
                (prediction.shape[0],),
                prediction.shape[1],
                dtype=torch.int64,
                device=prediction.device,
            )
        if reference_lengths is None:
            reference_lengths = torch.full(
                (reference.shape[0],),
                reference.shape[1],
                dtype=torch.int64,
                device=reference.device,
            )
        source = PointCloud.from_padded(prediction, prediction_lengths)
        target = PointCloud.from_padded(reference, reference_lengths)
    directional = "source_to_target" if single_directional else "both"
    return chamfer_distance(
        source,
        target,
        squared=squared,
        directional=directional,
        reduction=reduction,
    )


__all__ = [
    "FScoreResult",
    "PointCloudMetrics",
    "chamfer_distance",
    "chamfer_loss",
    "directed_nearest_distance",
    "fscore",
    "point_cloud_metrics",
]
