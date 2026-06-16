"""Point cloud comparison metrics (accuracy / completion / chamfer / F1).

Conventions (matching the MVS / surface-reconstruction literature):

- ``accuracy``: mean distance from each *prediction* point to its nearest
  *reference* point (how close the prediction lies to the reference).
- ``completion``: mean distance from each *reference* point to its
  nearest *prediction* point (how much of the reference is covered).
- ``chamfer_distance``: ``accuracy + completion`` (sum of the two
  directed means, unsquared L2).
- ``precision``: fraction of prediction points within ``threshold`` of
  the reference.
- ``recall``: fraction of reference points within ``threshold`` of the
  prediction.
- ``f1_score``: harmonic mean of precision and recall.

Nearest-neighbor distances are exact and unbounded by any search radius,
using the cuBQL-backed search extension.
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


def _nearest_distances(queries: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Exact distance from each query to its nearest point (unbounded)."""

    # Unbounded BVH search returns the NN index; the distance is then
    # recomputed in the input dtype. (The search itself runs in float32,
    # so for float64 inputs an eps-close tie may pick a different but
    # equidistant neighbor.)
    indices, _ = NearestNeighborSearch(points, math.inf).query(queries)
    return (queries - points[indices]).norm(dim=1)


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
        chamfer_distance=accuracy + completion,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )
