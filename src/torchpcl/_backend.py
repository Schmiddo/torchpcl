"""Private adapters for the consolidated native extension."""

from __future__ import annotations

import torch

from . import _C
from .cloud import PointCloud, batch_ids


def _check_device(device: torch.device) -> None:
    if device.type == "cpu":
        return
    if device.type == "cuda":
        if not _C.has_cuda():
            raise RuntimeError(
                "torchpcl was built without CUDA support; reinstall with "
                "TORCHPCL_WITH_CUDA=1 and a CUDA toolkit"
            )
        return
    raise RuntimeError(f"unsupported device '{device}' for torchpcl search")


def symmetric_eigh_3x3(
    matrices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted eigenvalues and the smallest eigenvector of symmetric 3x3s."""
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError("matrices must have shape (..., 3, 3)")
    return _C.symmetric_eigh_3x3(matrices.contiguous())


def farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Return num_samples row indices in greedy farthest-point selection order."""
    _check_device(points.device)
    return _C.farthest_point_sample(points.detach().contiguous(), num_samples)


def feature_knn1(
    reference: torch.Tensor, queries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact nearest feature row and squared distance per query."""
    _check_device(reference.device)
    return _C.feature_knn1(
        reference.detach().contiguous(), queries.detach().contiguous()
    )


def feature_mutual_knn1(
    source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nearest target rows of source rows and vice versa, with distances."""
    _check_device(source.device)
    return _C.feature_mutual_knn1(
        source.detach().contiguous(), target.detach().contiguous()
    )


def neighbor_moments(
    points: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return neighborhood means, centered scatter matrices and counts."""
    _check_device(points.device)
    return _C.neighbor_moments(points.detach().contiguous(), indices.contiguous())


def fpfh(
    points: torch.Tensor,
    normals: torch.Tensor,
    normal_valid: torch.Tensor,
    indices: torch.Tensor,
    distances2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FPFH descriptors and validity from a self-neighborhood."""
    _check_device(points.device)
    return _C.fpfh(
        points.detach().contiguous(),
        normals.detach().contiguous(),
        normal_valid.contiguous(),
        indices.contiguous(),
        distances2.detach().contiguous(),
    )


# Both backends return up to k nearest points within an inclusive radius, as
# int64 indices and exact float32 squared distances, with -1 and inf in empty
# slots (see csrc/backend.h).


class BvhBackend:
    def __init__(self, reference: PointCloud) -> None:
        _check_device(reference.device)
        self._points = reference.points.detach().to(torch.float32).contiguous()
        # (nodes, prim_ids), on the points' device (see csrc/backend.h).
        self._bvh = None if self._points.shape[0] == 0 else _C.build_bvh(self._points)

    def knn(
        self,
        queries: PointCloud,
        k: int,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_points = queries.points.detach().to(torch.float32).contiguous()
        if self._bvh is None:
            shape = (query_points.shape[0], k)
            return (
                torch.full(shape, -1, dtype=torch.int64, device=query_points.device),
                torch.full(
                    shape, torch.inf, dtype=torch.float32, device=query_points.device
                ),
            )
        return _C.bvh_knn(*self._bvh, self._points, query_points, k, radius)


class PackedBackend:
    def __init__(self, reference: PointCloud) -> None:
        _check_device(reference.device)
        self._points = reference.points.detach().to(torch.float32).contiguous()
        self._offsets = reference.offsets.contiguous()

    def knn(
        self,
        queries: PointCloud,
        k: int,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_points = queries.points.detach().to(torch.float32).contiguous()
        query_batch_ids = batch_ids(queries.offsets, query_points.shape[0])
        return _C.packed_knn(
            self._points,
            self._offsets,
            query_points,
            query_batch_ids.contiguous(),
            k,
            radius,
        )


__all__ = [
    "BvhBackend",
    "PackedBackend",
    "farthest_point_sample",
    "feature_knn1",
    "feature_mutual_knn1",
    "fpfh",
    "neighbor_moments",
    "symmetric_eigh_3x3",
]
