"""Point cloud comparison metrics (accuracy / completion / chamfer / F1).

Conventions (matching the MVS / surface-reconstruction literature):

- ``accuracy``: mean distance from each *prediction* point to its nearest
  *reference* point (how close the prediction lies to the reference).
- ``completion``: mean distance from each *reference* point to its
  nearest *prediction* point (how much of the reference is covered).
- ``chamfer_distance``: mean of ``accuracy`` and ``completion`` (average
  of the two directed means, unsquared L2).
- ``precision``: fraction of prediction points within ``threshold`` of
  the reference.
- ``recall``: fraction of reference points within ``threshold`` of the
  prediction.
- ``f1_score``: harmonic mean of precision and recall.

Nearest-neighbor distances are exact and unbounded by any search radius,
using the cuBQL-backed search extension.

Besides the evaluation-oriented ``point_cloud_metrics``, this module
provides ``chamfer_loss``, a differentiable (optionally batched) chamfer
loss for training.
"""

import math
from dataclasses import dataclass

import torch

from .search import NearestNeighborSearch
from .transforms import _check_points


@dataclass(frozen=True)
class PointCloudMetrics:
    accuracy: float
    completion: float
    chamfer_distance: float
    precision: float
    recall: float
    f1_score: float


def _nearest_squared_distances(queries: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Squared distance from each query to its nearest point (unbounded).

    Differentiable w.r.t. both tensors: the BVH search only provides the
    nearest-neighbor index (piecewise constant in the inputs, so treated
    as such by autograd); the distance is recomputed with torch ops in
    the input dtype. (The search itself runs in float32, so for float64
    inputs an eps-close tie may pick a different but equidistant
    neighbor.)
    """
    indices, _ = NearestNeighborSearch(points.detach(), math.inf).query(queries.detach())
    return (queries - points[indices]).square().sum(dim=1)


def _nearest_distances(queries: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Exact distance from each query to its nearest point (unbounded)."""
    return _nearest_squared_distances(queries, points).sqrt()


def _safe_sqrt(d2: torch.Tensor) -> torch.Tensor:
    """sqrt with a zero (instead of NaN) gradient at exactly zero.

    Zero squared distances occur whenever a query coincides with its
    nearest neighbor (e.g. identical clouds); plain sqrt would poison
    the whole backward pass with NaNs there. Values are exact.
    """
    positive = d2 > 0
    return torch.where(positive, d2, torch.ones_like(d2)).sqrt() * positive


def point_cloud_metrics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    threshold: float,
) -> PointCloudMetrics:
    """Compare a predicted/reconstructed cloud against a reference cloud.

    Args:
        prediction: (N, 3) predicted / reconstructed points.
        reference: (M, 3) reference (ground-truth) points, same device.
        threshold: Inlier distance for precision/recall/F1.

    Returns:
        PointCloudMetrics; distances are in the input units.
    """
    _check_points(prediction, "prediction")
    _check_points(reference, "reference")
    if len(prediction) == 0 or len(reference) == 0:
        raise ValueError("prediction and reference must be non-empty")
    if prediction.device != reference.device:
        raise ValueError("prediction and reference must be on the same device")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    reference = reference.to(prediction.dtype)

    dist_pred = _nearest_distances(prediction, reference)
    dist_ref = _nearest_distances(reference, prediction)

    accuracy = float(dist_pred.mean())
    completion = float(dist_ref.mean())
    precision = float((dist_pred <= threshold).to(torch.float64).mean())
    recall = float((dist_ref <= threshold).to(torch.float64).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return PointCloudMetrics(
        accuracy=accuracy,
        completion=completion,
        chamfer_distance=(accuracy + completion) / 2,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def _check_cloud(points: torch.Tensor, name: str) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim not in (2, 3) or points.shape[-1] != 3:
        raise ValueError(
            f"{name} must have shape (N, 3) or (B, N, 3), got {tuple(points.shape)}"
        )
    if not points.dtype.is_floating_point:
        raise ValueError(f"{name} must have a floating point dtype, got {points.dtype}")


def _check_lengths(
    lengths: torch.Tensor | None, name: str, points: torch.Tensor
) -> list[int] | None:
    if lengths is None:
        return None
    if (
        not isinstance(lengths, torch.Tensor)
        or lengths.ndim != 1
        or lengths.shape[0] != points.shape[0]
        or lengths.dtype.is_floating_point
    ):
        raise ValueError(f"{name} must be a (B,) integer tensor matching the batch size")
    values = lengths.tolist()
    if any(not 1 <= n <= points.shape[1] for n in values):
        raise ValueError(f"{name} entries must be in [1, {points.shape[1]}]")
    return values


def _chamfer_pair(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    squared: bool,
    single_directional: bool,
) -> torch.Tensor:
    if len(prediction) == 0 or len(reference) == 0:
        raise ValueError("prediction and reference must be non-empty")

    def directed(queries: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        d2 = _nearest_squared_distances(queries, points)
        return d2.mean() if squared else _safe_sqrt(d2).mean()

    loss = directed(prediction, reference)
    if not single_directional:
        loss = (loss + directed(reference, prediction)) / 2
    return loss


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
    """Differentiable chamfer loss between two point clouds.

    Per cloud pair: the mean (squared) distance from each prediction
    point to its nearest reference point, averaged with the reverse
    direction unless ``single_directional``. Gradients flow to both
    clouds. With ``squared=False`` the unbatched bidirectional loss
    equals ``point_cloud_metrics(...).chamfer_distance``.

    Args:
        prediction: (N, 3) or padded (B, N, 3) points.
        reference: (M, 3) or padded (B, M, 3) points, same device.
        prediction_lengths: (B,) valid point counts per batch entry
            (batched input only); padding beyond each length is ignored.
        reference_lengths: Same for the reference clouds.
        squared: Use squared L2 distances (default) instead of L2.
        single_directional: Only the prediction -> reference direction.
        reduction: "mean" | "sum" | "none" over the batch dimension.

    Returns:
        A 0-dim tensor for (N, 3) inputs; for batched inputs a 0-dim
        tensor, or (B,) with ``reduction="none"``.
    """
    _check_cloud(prediction, "prediction")
    _check_cloud(reference, "reference")
    if prediction.ndim != reference.ndim:
        raise ValueError("prediction and reference must both be batched or both unbatched")
    if prediction.device != reference.device:
        raise ValueError("prediction and reference must be on the same device")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got '{reduction}'")
    reference = reference.to(prediction.dtype)

    if prediction.ndim == 2:
        if prediction_lengths is not None or reference_lengths is not None:
            raise ValueError("lengths are only valid for batched (B, N, 3) inputs")
        return _chamfer_pair(prediction, reference, squared, single_directional)

    if prediction.shape[0] != reference.shape[0]:
        raise ValueError("prediction and reference must have the same batch size")
    pred_lengths = _check_lengths(prediction_lengths, "prediction_lengths", prediction)
    ref_lengths = _check_lengths(reference_lengths, "reference_lengths", reference)
    losses = torch.stack([
        _chamfer_pair(
            prediction[b, : pred_lengths[b]] if pred_lengths else prediction[b],
            reference[b, : ref_lengths[b]] if ref_lengths else reference[b],
            squared,
            single_directional,
        )
        for b in range(prediction.shape[0])
    ])
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    return losses
