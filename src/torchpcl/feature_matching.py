"""Correspondence construction from point-aligned feature tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._backend import feature_knn1
from .cloud import PointCloud
from .validation import check_cloud_pair


@dataclass(frozen=True, eq=False)
class CorrespondenceSet:
    """Point-row correspondences for a single source and target cloud."""

    source_indices: torch.Tensor
    target_indices: torch.Tensor
    distances2: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_indices, torch.Tensor) or not isinstance(
            self.target_indices, torch.Tensor
        ):
            raise TypeError("correspondence indices must be torch.Tensor objects")
        if (
            self.source_indices.ndim != 1
            or self.target_indices.shape != self.source_indices.shape
        ):
            raise ValueError("source_indices and target_indices must have shape (C,)")
        if (
            self.source_indices.dtype != torch.int64
            or self.target_indices.dtype != torch.int64
        ):
            raise ValueError("correspondence indices must have dtype int64")
        if self.source_indices.device != self.target_indices.device:
            raise ValueError("correspondence indices must be on the same device")
        if self.distances2 is not None:
            if not isinstance(self.distances2, torch.Tensor):
                raise TypeError("distances2 must be a torch.Tensor or None")
            if self.distances2.shape != self.source_indices.shape:
                raise ValueError("distances2 must have shape (C,)")
            if self.distances2.device != self.source_indices.device:
                raise ValueError("distances2 and indices must be on the same device")

    @property
    def count(self) -> int:
        return self.source_indices.shape[0]


def _features(cloud: PointCloud, name: str) -> torch.Tensor:
    if cloud.features is None:
        raise ValueError(f"{name} must have features attached")
    if cloud.features.ndim != 2:
        raise ValueError(f"{name} features must have shape (P, D)")
    if cloud.features.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} features must have dtype float32 or float64")
    if not bool(torch.isfinite(cloud.features).all()):
        raise ValueError(f"{name} features must be finite")
    return cloud.features


@torch.no_grad()
def match_features(
    source: PointCloud,
    target: PointCloud,
    *,
    mutual: bool = True,
) -> CorrespondenceSet:
    """Match point-aligned features using exact squared L2 distance.

    This MVP implementation supports one source and target cloud. The exact
    brute-force search streams over candidates without materializing the full
    descriptor distance matrix.
    """
    if not isinstance(source, PointCloud) or not isinstance(target, PointCloud):
        raise TypeError("source and target must be PointCloud objects")
    if not isinstance(mutual, bool):
        raise TypeError("mutual must be a bool")
    check_cloud_pair(source, target, "source", "target")
    if source.batch_size != 1:
        raise ValueError("feature matching currently supports batch_size == 1")

    source_features = _features(source, "source")
    target_features = _features(target, "target")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    if source_features.dtype != target_features.dtype:
        raise ValueError("source and target features must have the same dtype")

    if source_features.shape[0] == 0 or target_features.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.int64, device=source.device)
        return CorrespondenceSet(
            source_indices=empty,
            target_indices=empty.clone(),
            distances2=source_features.new_empty(0),
        )

    target_indices, distances2 = feature_knn1(target_features, source_features)
    source_indices = torch.arange(
        source_features.shape[0], dtype=torch.int64, device=source.device
    )
    if mutual:
        reverse, _ = feature_knn1(source_features, target_features)
        keep = reverse[target_indices] == source_indices
        source_indices = source_indices[keep]
        target_indices = target_indices[keep]
        distances2 = distances2[keep]

    return CorrespondenceSet(source_indices, target_indices, distances2)


__all__ = ["CorrespondenceSet", "match_features"]
