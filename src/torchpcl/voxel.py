"""Packed voxelization and reusable per-voxel reductions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cloud import PointCloud, batch_ids


def _as_cloud(value: torch.Tensor | PointCloud) -> tuple[PointCloud, bool]:
    if isinstance(value, PointCloud):
        return value, False
    if isinstance(value, torch.Tensor):
        return PointCloud.from_points(value), True
    raise TypeError("cloud must be a torch.Tensor or PointCloud")


@dataclass(frozen=True, eq=False)
class Voxelization:
    """A voxelized cloud and the mapping from input points to voxels."""

    cloud: PointCloud
    coordinates: torch.Tensor
    point_to_voxel: torch.Tensor
    counts: torch.Tensor

    def reduce(self, values: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduce point-aligned values using the existing voxel partition."""
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if values.ndim < 1 or values.shape[0] != self.point_to_voxel.shape[0]:
            raise ValueError(
                "values must have the same leading dimension as the input points"
            )
        if values.device != self.point_to_voxel.device:
            raise ValueError("values and voxelization must be on the same device")
        if reduction not in {"mean", "sum", "min", "max", "first"}:
            raise ValueError("reduction must be mean, sum, min, max, or first")

        output_shape = (self.counts.shape[0], *values.shape[1:])
        if self.counts.shape[0] == 0:
            return values.new_empty(output_shape)
        flat = values.reshape(values.shape[0], -1)
        columns = flat.shape[1]
        indices = self.point_to_voxel[:, None].expand(-1, columns)

        if reduction in {"sum", "mean"}:
            reduced = values.new_zeros((self.counts.shape[0], columns))
            reduced.scatter_add_(0, indices, flat)
            if reduction == "mean":
                reduced = reduced / self.counts.to(values.dtype)[:, None]
        elif reduction in {"min", "max"}:
            reduce_name = "amin" if reduction == "min" else "amax"
            reduced = values.new_zeros((self.counts.shape[0], columns))
            reduced.scatter_reduce_(
                0, indices, flat, reduce=reduce_name, include_self=False
            )
        else:
            rows = torch.arange(
                values.shape[0], dtype=torch.int64, device=values.device
            )
            first = torch.full(
                (self.counts.shape[0],),
                values.shape[0],
                dtype=torch.int64,
                device=values.device,
            )
            first.scatter_reduce_(
                0, self.point_to_voxel, rows, reduce="amin", include_self=True
            )
            reduced = flat[first]
        return reduced.reshape(output_shape)


def voxelize(
    cloud: torch.Tensor | PointCloud,
    voxel_size: float,
) -> Voxelization:
    """Group packed points by ``floor(point / voxel_size)`` and average them."""
    packed, _ = _as_cloud(cloud)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    point_batches = batch_ids(packed.offsets, packed.points.shape[0])
    spatial = torch.floor(packed.points / voxel_size).to(torch.int64)
    point_coordinates = torch.cat([point_batches[:, None], spatial], dim=1)

    # torch.unique(dim=0) compares rows through a particularly slow CPU path.
    # Four stable scalar sorts give the same lexicographic order while keeping
    # signed int64 coordinates unrestricted.
    order = torch.arange(point_coordinates.shape[0], device=packed.device)
    for column in range(3, -1, -1):
        column_order = point_coordinates[order, column].argsort(stable=True)
        order = order[column_order]
    sorted_coordinates = point_coordinates[order]
    starts = torch.ones(
        sorted_coordinates.shape[0], dtype=torch.bool, device=packed.device
    )
    if sorted_coordinates.shape[0] > 1:
        starts[1:] = torch.any(
            sorted_coordinates[1:] != sorted_coordinates[:-1], dim=1
        )
    sorted_voxel_ids = starts.cumsum(0) - 1
    inverse = torch.empty_like(sorted_voxel_ids)
    inverse[order] = sorted_voxel_ids
    coordinates = sorted_coordinates[starts]
    counts = torch.bincount(sorted_voxel_ids, minlength=coordinates.shape[0])

    voxel_count = coordinates.shape[0]
    sums = packed.points.new_zeros((voxel_count, 3))
    sums.index_add_(0, inverse, packed.points)
    centroids = sums / counts.to(packed.dtype)[:, None]

    if voxel_count:
        per_batch = torch.bincount(
            coordinates[:, 0], minlength=packed.batch_size
        )
    else:
        per_batch = torch.zeros(
            packed.batch_size, dtype=torch.int64, device=packed.device
        )
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=packed.device),
            per_batch.cumsum(0),
        ]
    )
    voxel_cloud = PointCloud._from_validated(centroids, offsets)
    return Voxelization(
        cloud=voxel_cloud,
        coordinates=coordinates,
        point_to_voxel=inverse,
        counts=counts,
    )


__all__ = ["Voxelization", "voxelize"]
