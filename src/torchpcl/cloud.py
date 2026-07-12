"""Packed point-cloud data model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .validation import check_geometry_tensor, check_offsets, check_point_attribute


@dataclass(frozen=True, eq=False)
class PointCloud:
    """A packed ragged batch of point clouds.

    ``points`` stores all points contiguously by batch and ``offsets`` delimits
    each batch element. Tensor storage is shared by construction and by no-op
    calls to :meth:`to`; use :meth:`clone` for independent storage. The frozen
    dataclass prevents field reassignment, but does not make the tensors
    themselves immutable.

    Floating geometry is restricted to float32 and float64. Features retain
    their own dtype when the cloud geometry is converted with :meth:`to`.
    """

    points: torch.Tensor
    offsets: torch.Tensor
    normals: torch.Tensor | None = None
    features: torch.Tensor | None = None

    def __post_init__(self) -> None:
        check_geometry_tensor(self.points, "points")
        check_offsets(self.offsets, self.points)
        check_point_attribute(self.normals, self.points, "normals", exact_shape=True)
        if self.normals is not None and self.normals.dtype != self.points.dtype:
            raise ValueError("normals and points must have the same dtype")
        check_point_attribute(self.features, self.points, "features")

    @classmethod
    def _from_validated(
        cls,
        points: torch.Tensor,
        offsets: torch.Tensor,
        normals: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> PointCloud:
        """Construct from tensors derived from an already validated cloud."""
        cloud = object.__new__(cls)
        object.__setattr__(cloud, "points", points)
        object.__setattr__(cloud, "offsets", offsets)
        object.__setattr__(cloud, "normals", normals)
        object.__setattr__(cloud, "features", features)
        return cloud

    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,
        *,
        normals: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> PointCloud:
        """Create a one-element batch from an ``(N, 3)`` tensor."""
        check_geometry_tensor(points, "points")
        offsets = torch.tensor(
            [0, points.shape[0]], dtype=torch.int64, device=points.device
        )
        check_point_attribute(normals, points, "normals", exact_shape=True)
        if normals is not None and normals.dtype != points.dtype:
            raise ValueError("normals and points must have the same dtype")
        check_point_attribute(features, points, "features")
        return cls._from_validated(points, offsets, normals, features)

    @classmethod
    def from_padded(
        cls,
        points: torch.Tensor,
        lengths: torch.Tensor,
        *,
        normals: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> PointCloud:
        """Pack a padded ``(B, N, 3)`` batch using per-cloud lengths."""
        if not isinstance(points, torch.Tensor):
            raise TypeError(
                f"points must be a torch.Tensor, got {type(points).__name__}"
            )
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(
                f"points must have shape (B, N, 3), got {tuple(points.shape)}"
            )
        if points.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"points must have dtype float32 or float64, got {points.dtype}"
            )
        if not isinstance(lengths, torch.Tensor):
            raise TypeError(
                f"lengths must be a torch.Tensor, got {type(lengths).__name__}"
            )
        if lengths.ndim != 1 or lengths.shape[0] != points.shape[0]:
            raise ValueError(
                "lengths must have shape (B,) matching the padded batch"
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if lengths.dtype not in integer_dtypes:
            raise ValueError("lengths must have an integer dtype")

        lengths = lengths.to(device=points.device, dtype=torch.int64)
        lengths_valid = torch.all((lengths >= 0) & (lengths <= points.shape[1]))
        if not bool(lengths_valid):
            raise ValueError(f"lengths entries must be in [0, {points.shape[1]}]")

        row = torch.arange(points.shape[1], device=points.device)
        mask = row.unsqueeze(0) < lengths.unsqueeze(1)
        packed_points = points[mask]
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=points.device), lengths.cumsum(0)]
        )

        packed_normals = cls._pack_attribute(
            normals, mask, points, "normals", exact_shape=True
        )
        packed_features = cls._pack_attribute(
            features, mask, points, "features", exact_shape=False
        )
        return cls._from_validated(
            packed_points, offsets, packed_normals, packed_features
        )

    @staticmethod
    def _pack_attribute(
        values: torch.Tensor | None,
        mask: torch.Tensor,
        points: torch.Tensor,
        name: str,
        *,
        exact_shape: bool,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        shape_valid = tuple(values.shape) == tuple(points.shape) if exact_shape else (
            values.ndim >= 2 and values.shape[:2] == points.shape[:2]
        )
        if not shape_valid:
            expected = tuple(points.shape) if exact_shape else (
                points.shape[0],
                points.shape[1],
                "...",
            )
            raise ValueError(f"{name} must have shape {expected}, got {tuple(values.shape)}")
        if values.device != points.device:
            raise ValueError(f"{name} and points must be on the same device")
        if name == "normals" and values.dtype != points.dtype:
            raise ValueError("normals and points must have the same dtype")
        return values[mask]

    @property
    def batch_size(self) -> int:
        return self.offsets.shape[0] - 1

    @property
    def lengths(self) -> torch.Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    @property
    def device(self) -> torch.device:
        return self.points.device

    @property
    def dtype(self) -> torch.dtype:
        return self.points.dtype

    def to_padded(self, pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded points ``(B, N, 3)`` and their lengths.

        ``N`` is the longest batch length. Attached normals and features remain
        on the cloud; callers can gather them separately when required.
        """
        lengths = self.lengths
        max_length = int(lengths.max().item()) if self.batch_size else 0
        padded = torch.full(
            (self.batch_size, max_length, 3),
            pad_value,
            dtype=self.dtype,
            device=self.device,
        )
        if self.points.shape[0] > 0:
            ids = batch_ids(self.offsets, self.points.shape[0])
            local_rows = torch.arange(self.points.shape[0], device=self.device)
            local_rows = local_rows - self.offsets[ids]
            padded[ids, local_rows] = self.points
        return padded, lengths

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> PointCloud:
        """Move or cast geometry while preserving the feature dtype."""
        if isinstance(device, torch.dtype):
            if dtype is not None:
                raise TypeError("dtype was specified twice")
            dtype = device
            device = None
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be float32 or float64, got {dtype}")

        points = self.points.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        offsets = self.offsets.to(
            device=device,
            non_blocking=non_blocking,
            copy=copy,
        )
        normals = None if self.normals is None else self.normals.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        features = None if self.features is None else self.features.to(
            device=device,
            non_blocking=non_blocking,
            copy=copy,
        )
        return self._from_validated(points, offsets, normals, features)

    def clone(self) -> PointCloud:
        """Clone all attached tensor storage."""
        return self._from_validated(
            self.points.clone(),
            self.offsets.clone(),
            None if self.normals is None else self.normals.clone(),
            None if self.features is None else self.features.clone(),
        )


def batch_ids(offsets: torch.Tensor, total_size: int) -> torch.Tensor:
    """Return the batch index for every row represented by ``offsets``."""
    lengths = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=offsets.device),
        lengths,
        output_size=total_size,
    )


__all__ = ["PointCloud"]
