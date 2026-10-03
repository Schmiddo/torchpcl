"""Packed voxelization and reusable per-voxel reductions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cloud import (
    PointCloud,
    PointCloudLike,
    _normalize_cloud,
    _pack_aligned_shape,
    batch_ids,
)


@dataclass(frozen=True, eq=False)
class Voxelization:
    """A voxelized cloud and the mapping from input points to voxels."""

    cloud: PointCloud
    coordinates: torch.Tensor
    point_to_voxel: torch.Tensor
    counts: torch.Tensor
    _input_leading_shape: tuple[int, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._input_leading_shape is None:
            object.__setattr__(
                self, "_input_leading_shape", (self.point_to_voxel.shape[0],)
            )

    def reduce(self, values: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduce point-aligned values using the existing voxel partition."""
        leading_shape = self._input_leading_shape
        assert leading_shape is not None
        values = _pack_aligned_shape(
            values,
            leading_shape,
            self.point_to_voxel.shape[0],
            self.point_to_voxel.device,
            "values",
        )
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
    cloud: PointCloudLike,
    voxel_size: float,
) -> Voxelization:
    """Group packed points by ``floor(point / voxel_size)`` and average them."""
    normalized = _normalize_cloud(cloud)
    packed = normalized.cloud
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    point_batches = batch_ids(packed.offsets, packed.points.shape[0]).to(torch.int32)
    spatial = torch.floor(packed.points / voxel_size).to(torch.int32)
    point_coordinates = torch.cat([point_batches[:, None], spatial], dim=1)
    del point_batches, spatial

    # torch.unique(dim=0) compares rows through a particularly slow CPU path.
    # Four stable scalar sorts give the same lexicographic order while keeping
    # signed int64 coordinates unrestricted.
    order = torch.arange(point_coordinates.shape[0], device=packed.device)
    for column in range(3, -1, -1):
        column_order = point_coordinates[order, column].argsort(stable=True)
        order = order[column_order]
    sorted_coordinates = point_coordinates[order]
    point_count = sorted_coordinates.shape[0]
    starts = torch.ones(point_count, dtype=torch.bool, device=packed.device)
    if point_count > 1:
        starts[1:] = torch.any(
            sorted_coordinates[1:] != sorted_coordinates[:-1], dim=1
        )
    start_rows = starts.nonzero()[:, 0]
    sorted_voxel_ids = starts.cumsum(0) - 1
    inverse = torch.empty_like(sorted_voxel_ids)
    inverse[order] = sorted_voxel_ids
    coordinates = sorted_coordinates[start_rows]
    counts = torch.diff(start_rows, append=start_rows.new_tensor([point_count]))

    # Each voxel's points are contiguous in sorted order, so the centroids
    # are segment sums: deterministic on CUDA, unlike an atomic scatter-add,
    # and faster on the CPU, where scatter-add runs on one thread.
    if point_count:
        sums = torch.segment_reduce(
            packed.points[order], "sum", lengths=counts, axis=0
        )
    else:
        sums = packed.points.new_zeros((0, 3))
    centroids = sums / counts.to(packed.dtype)[:, None]

    # Voxels are sorted by batch first, so each batch's are contiguous.
    offsets = torch.searchsorted(
        coordinates[:, 0].contiguous(),
        torch.arange(
            packed.batch_size + 1, dtype=torch.int32, device=packed.device
        ),
    )
    voxel_cloud = PointCloud._from_validated(centroids, offsets)
    return Voxelization(
        cloud=voxel_cloud,
        coordinates=coordinates,
        point_to_voxel=inverse,
        counts=counts,
        _input_leading_shape=normalized.leading_shape,
    )


__all__ = ["Voxelization", "voxelize"]
