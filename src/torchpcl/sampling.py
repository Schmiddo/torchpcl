"""Farthest point sampling for a single packed point cloud."""

from __future__ import annotations

import torch

from .cloud import PointCloud, PointCloudLike, _normalize_cloud


@torch.no_grad()
def farthest_point_sample(cloud: PointCloudLike, num_samples: int) -> PointCloud:
    """Greedily select points that maximize the minimum pairwise distance.

    Only a single cloud is supported (a ``(N, 3)`` tensor, or a
    ``PointCloud``/``(B, N, 3)`` tensor with batch size 1); batching is not
    yet implemented. Selection starts from point index 0 and proceeds
    deterministically. Points are returned in selection order, not input
    order. If ``num_samples >= N``, all points are returned unchanged.
    """
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    normalized = _normalize_cloud(cloud)
    packed = normalized.cloud
    if packed.batch_size != 1:
        raise ValueError(
            "farthest_point_sample only supports a single cloud (batch size "
            f"1), got batch size {packed.batch_size}"
        )

    points = packed.points
    num_points = points.shape[0]
    if num_samples >= num_points:
        return packed

    selected = torch.empty(num_samples, dtype=torch.int64, device=points.device)
    min_dist2 = (points - points[0]).square().sum(-1)
    selected[0] = 0
    for i in range(1, num_samples):
        next_index = int(min_dist2.argmax().item())
        selected[i] = next_index
        min_dist2[next_index] = -torch.inf
        dist2_to_new = (points - points[next_index]).square().sum(-1)
        min_dist2 = torch.minimum(min_dist2, dist2_to_new)

    normals = None if packed.normals is None else packed.normals[selected]
    features = None if packed.features is None else packed.features[selected]
    offsets = torch.tensor([0, num_samples], dtype=torch.int64, device=points.device)
    return PointCloud._from_validated(points[selected], offsets, normals, features)


__all__ = ["farthest_point_sample"]
