"""Shared validation helpers for the packed API."""

from __future__ import annotations

import torch

_GEOMETRY_DTYPES = (torch.float32, torch.float64)


def check_geometry_tensor(points: torch.Tensor, name: str) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (P, 3), got {tuple(points.shape)}")
    if points.dtype not in _GEOMETRY_DTYPES:
        raise ValueError(
            f"{name} must have dtype float32 or float64, got {points.dtype}"
        )


def check_offsets(offsets: torch.Tensor, points: torch.Tensor) -> None:
    if not isinstance(offsets, torch.Tensor):
        raise TypeError(
            f"offsets must be a torch.Tensor, got {type(offsets).__name__}"
        )
    if offsets.ndim != 1 or offsets.numel() < 1:
        raise ValueError(
            f"offsets must have shape (B + 1,), got {tuple(offsets.shape)}"
        )
    if offsets.dtype != torch.int64:
        raise ValueError(f"offsets must have dtype int64, got {offsets.dtype}")
    if offsets.device != points.device:
        raise ValueError("offsets and points must be on the same device")

    # Combining all value checks incurs at most one device synchronization for
    # user-provided CUDA offsets. Internally derived clouds bypass revalidation.
    valid = (
        (offsets[0] == 0)
        & (offsets[-1] == points.shape[0])
        & torch.all(offsets[1:] >= offsets[:-1])
    )
    if not bool(valid):
        raise ValueError(
            "offsets must start at zero, be nondecreasing, and end at len(points)"
        )


def check_point_attribute(
    values: torch.Tensor | None,
    points: torch.Tensor,
    name: str,
    *,
    exact_shape: bool = False,
) -> None:
    if values is None:
        return
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(values).__name__}")
    expected = tuple(points.shape)
    shape_valid = tuple(values.shape) == expected if exact_shape else (
        values.ndim >= 1 and values.shape[0] == points.shape[0]
    )
    if not shape_valid:
        requirement = f"shape {expected}" if exact_shape else f"leading dimension {points.shape[0]}"
        raise ValueError(f"{name} must have {requirement}, got {tuple(values.shape)}")
    if values.device != points.device:
        raise ValueError(f"{name} and points must be on the same device")


__all__ = ["check_geometry_tensor", "check_offsets", "check_point_attribute"]
