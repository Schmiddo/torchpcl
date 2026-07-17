"""Differentiable alignment of point clouds with known correspondences."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._segments import segment_sum
from .cloud import (
    PointCloudLike,
    _normalize_cloud,
    _pack_aligned,
    batch_ids,
)
from .validation import check_cloud_pair


@dataclass(frozen=True, eq=False)
class ProcrustesResult:
    """Batched orientation-preserving similarity transformations.

    The transformation maps a source point ``x`` to
    ``scale * (rotation @ x) + translation``. The homogeneous matrices are
    constructed on demand so gradients continue to flow to every component.
    """

    rotation: torch.Tensor
    translation: torch.Tensor
    scale: torch.Tensor

    @property
    def transforms(self) -> torch.Tensor:
        """Return homogeneous ``(B, 4, 4)`` similarity matrices."""
        linear = self.scale[:, None, None] * self.rotation
        upper = torch.cat([linear, self.translation[:, :, None]], dim=2)
        bottom = self.rotation.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(
            self.rotation.shape[0], 1, 4
        )
        return torch.cat([upper, bottom], dim=1)


def _procrustes_packed(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    *,
    estimate_scale: bool,
) -> tuple[ProcrustesResult, torch.Tensor]:
    """Solve packed weighted alignment without public-input validation.

    The singular values are returned for callers that need to diagnose
    degenerate correspondence sets. Zero-weight batches are permitted here so
    inference-only registration can mask inactive entries after solving.
    """
    batch_size = offsets.shape[0] - 1
    weight_sum = segment_sum(weights, offsets)
    safe_weight_sum = weight_sum.clamp_min(torch.finfo(weights.dtype).tiny)
    source_mean = segment_sum(source * weights[:, None], offsets)
    source_mean = source_mean / safe_weight_sum[:, None]
    target_mean = segment_sum(target * weights[:, None], offsets)
    target_mean = target_mean / safe_weight_sum[:, None]

    centered_source = source - source_mean[ids]
    centered_target = target - target_mean[ids]
    covariance = centered_source[:, :, None] * centered_target[:, None, :]
    covariance = segment_sum(covariance * weights[:, None, None], offsets)

    u, singular_values, vh = torch.linalg.svd(covariance)
    v = vh.transpose(1, 2)
    determinant = torch.linalg.det(v @ u.transpose(1, 2))
    diagonal = torch.stack(
        [torch.ones_like(determinant), torch.ones_like(determinant), determinant],
        dim=1,
    )
    correction = torch.diag_embed(diagonal)
    rotation = v @ correction @ u.transpose(1, 2)

    if estimate_scale:
        source_variance = segment_sum(
            weights * centered_source.square().sum(dim=1), offsets
        )
        scale = (singular_values * diagonal).sum(dim=1) / source_variance.clamp_min(
            torch.finfo(source.dtype).tiny
        )
    else:
        scale = torch.ones(batch_size, dtype=source.dtype, device=source.device)

    rotated_mean = (rotation @ source_mean[:, :, None]).squeeze(-1)
    translation = target_mean - scale[:, None] * rotated_mean
    return ProcrustesResult(rotation, translation, scale), singular_values


def procrustes(
    source: PointCloudLike,
    target: PointCloudLike,
    *,
    weights: torch.Tensor | None = None,
    estimate_scale: bool = False,
) -> ProcrustesResult:
    """Align corresponding source and target points with a differentiable SVD.

    Tensor inputs may have shape ``(N, 3)`` or ``(B, N, 3)``. Packed
    :class:`PointCloud` inputs support ragged batches; corresponding batch
    entries must have equal lengths and points correspond by packed row.

    ``weights`` optionally supplies one nonnegative weight per point and must
    match the source leading shape: ``(N,)``, ``(B, N)``, or packed ``(P,)``.
    With ``estimate_scale=False`` this solves weighted rigid Procrustes/Kabsch.
    With ``estimate_scale=True`` it solves the orientation-preserving Umeyama
    similarity problem. Results are always batched.

    The solution is differentiable for nondegenerate correspondence sets.
    Gradients are mathematically undefined where the optimal rotation is not
    locally unique, so batches with covariance rank below two are rejected.
    """
    if not isinstance(estimate_scale, bool):
        raise TypeError("estimate_scale must be a bool")
    normalized_source = _normalize_cloud(source, "source")
    normalized_target = _normalize_cloud(target, "target")
    source_cloud = normalized_source.cloud
    target_cloud = normalized_target.cloud
    check_cloud_pair(
        source_cloud,
        target_cloud,
        "source",
        "target",
        equal_lengths=True,
    )

    if weights is None:
        point_weights = source_cloud.points.new_ones(source_cloud.points.shape[0])
    else:
        point_weights = _pack_aligned(
            weights, normalized_source, "weights", trailing_shape=()
        )
        if (
            point_weights.device != source_cloud.device
            or point_weights.dtype != source_cloud.dtype
        ):
            raise ValueError("weights must match the source device and dtype")
        if bool(torch.any(~torch.isfinite(point_weights))) or bool(
            torch.any(point_weights < 0)
        ):
            raise ValueError("weights must be finite and nonnegative")

    ids = batch_ids(source_cloud.offsets, source_cloud.points.shape[0])
    positive_counts = segment_sum(
        (point_weights > 0).to(source_cloud.dtype), source_cloud.offsets
    )
    if bool(torch.any(positive_counts < 3)):
        raise ValueError("each batch entry needs at least 3 positively weighted points")

    result, singular_values = _procrustes_packed(
        source_cloud.points,
        target_cloud.points,
        point_weights,
        ids,
        source_cloud.offsets,
        estimate_scale=estimate_scale,
    )
    tolerance = singular_values[:, :1] * (
        3 * torch.finfo(singular_values.dtype).eps
    )
    rank = (singular_values > tolerance).sum(dim=1)
    if bool(torch.any(rank < 2)):
        raise ValueError("each correspondence set must be non-collinear")
    if not bool(torch.isfinite(result.transforms).all()):
        raise ValueError("alignment produced a non-finite transformation")
    return result


__all__ = ["ProcrustesResult", "procrustes"]
