"""Private adapter for the consolidated native search extension."""

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


class BvhBackend:
    def __init__(self, reference: PointCloud) -> None:
        _check_device(reference.device)
        self._points = reference.points.detach().to(torch.float32).contiguous()
        self._index = None if self._points.shape[0] == 0 else _C.BvhIndex(self._points)

    def knn(
        self,
        queries: PointCloud,
        k: int,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_points = queries.points.detach().to(torch.float32).contiguous()
        if self._index is None:
            shape = (query_points.shape[0], k)
            return (
                torch.full(shape, -1, dtype=torch.int32, device=query_points.device),
                torch.full(
                    shape,
                    float(radius * radius),
                    dtype=torch.float32,
                    device=query_points.device,
                ),
            )
        return self._index.knn(query_points, k, radius)


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


__all__ = ["BvhBackend", "PackedBackend"]
