"""Differentiable alignment of point clouds with known correspondences."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._segments import segment_sum
from .cloud import PointCloud, as_cloud, batch_ids


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
    batch_size: int,
    *,
    estimate_scale: bool,
) -> tuple[ProcrustesResult, torch.Tensor]:
    """Solve packed weighted alignment without public-input validation.

    The singular values are returned for callers that need to diagnose
    degenerate correspondence sets. Zero-weight batches are permitted here so
    inference-only registration can mask inactive entries after solving.
    """
    weight_sum = segment_sum(weights, ids, batch_size)
    safe_weight_sum = weight_sum.clamp_min(torch.finfo(weights.dtype).tiny)
    source_mean = segment_sum(source * weights[:, None], ids, batch_size)
    source_mean = source_mean / safe_weight_sum[:, None]
    target_mean = segment_sum(target * weights[:, None], ids, batch_size)
    target_mean = target_mean / safe_weight_sum[:, None]

    centered_source = source - source_mean[ids]
    centered_target = target - target_mean[ids]
    covariance = centered_source[:, :, None] * centered_target[:, None, :]
    covariance = segment_sum(
        covariance * weights[:, None, None], ids, batch_size
    )

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
            weights * centered_source.square().sum(dim=1), ids, batch_size
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
    source: torch.Tensor | PointCloud,
    target: torch.Tensor | PointCloud,
    *,
    weights: torch.Tensor | None = None,
    estimate_scale: bool = False,
) -> ProcrustesResult:
    """Align corresponding source and target points with a differentiable SVD.

    Tensor inputs represent one cloud and must have shape ``(N, 3)``. Packed
    :class:`PointCloud` inputs support ragged batches; corresponding batch
    entries must have equal lengths and points correspond by packed row.

    ``weights`` optionally supplies one nonnegative weight per packed point.
    With ``estimate_scale=False`` this solves weighted rigid Procrustes/Kabsch.
    With ``estimate_scale=True`` it solves the orientation-preserving Umeyama
    similarity problem. Results are always batched.

    The solution is differentiable for nondegenerate correspondence sets.
    Gradients are mathematically undefined where the optimal rotation is not
    locally unique, so batches with covariance rank below two are rejected.
    """
    if not isinstance(estimate_scale, bool):
        raise TypeError("estimate_scale must be a bool")
    source_cloud = as_cloud(source, "source")
    target_cloud = as_cloud(target, "target")
    if source_cloud.batch_size != target_cloud.batch_size:
        raise ValueError("source and target must have the same batch size")
    if source_cloud.device != target_cloud.device:
        raise ValueError("source and target must be on the same device")
    if source_cloud.dtype != target_cloud.dtype:
        raise ValueError("source and target must have the same dtype")
    if not bool(torch.equal(source_cloud.lengths, target_cloud.lengths)):
        raise ValueError("source and target batch entries must have equal lengths")

    if weights is None:
        point_weights = source_cloud.points.new_ones(source_cloud.points.shape[0])
    else:
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor")
        if weights.shape != (source_cloud.points.shape[0],):
            raise ValueError("weights must have shape (P,)")
        if (
            weights.device != source_cloud.device
            or weights.dtype != source_cloud.dtype
        ):
            raise ValueError("weights must match the source device and dtype")
        if bool(torch.any(~torch.isfinite(weights))) or bool(torch.any(weights < 0)):
            raise ValueError("weights must be finite and nonnegative")
        point_weights = weights

    ids = batch_ids(source_cloud.offsets, source_cloud.points.shape[0])
    positive_counts = segment_sum(
        (point_weights > 0).to(torch.int64), ids, source_cloud.batch_size
    )
    if bool(torch.any(positive_counts < 3)):
        raise ValueError("each batch entry needs at least 3 positively weighted points")

    result, singular_values = _procrustes_packed(
        source_cloud.points,
        target_cloud.points,
        point_weights,
        ids,
        source_cloud.batch_size,
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
