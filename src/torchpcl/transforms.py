"""Rigid transformation helpers (pure torch, float64)."""

import torch


def _check_points(points: torch.Tensor, name: str) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(points.shape)}")
    if not points.dtype.is_floating_point:
        raise ValueError(f"{name} must have a floating point dtype, got {points.dtype}")


def transform_points(points: torch.Tensor, transformation: torch.Tensor) -> torch.Tensor:
    """Apply a (4, 4) rigid transformation to (N, 3) points."""
    rotation = transformation[:3, :3].to(points.dtype)
    translation = transformation[:3, 3].to(points.dtype)
    return points @ rotation.T + translation


def pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """Convert a 6-vector pose [rx, ry, rz, tx, ty, tz] to a (4, 4) matrix.

    The rotation is composed as R = Rz(rz) @ Ry(ry) @ Rx(rx), matching
    Open3D's PoseToTransformationImpl (ZYX euler angles).
    """
    pose = pose.to(torch.float64)
    rx, ry, rz = pose[0], pose[1], pose[2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    transformation = torch.eye(4, dtype=torch.float64, device=pose.device)
    transformation[0, 0] = cz * cy
    transformation[0, 1] = -sz * cx + cz * sy * sx
    transformation[0, 2] = sz * sx + cz * sy * cx
    transformation[1, 0] = sz * cy
    transformation[1, 1] = cz * cx + sz * sy * sx
    transformation[1, 2] = -cz * sx + sz * sy * cx
    transformation[2, 0] = -sy
    transformation[2, 1] = cy * sx
    transformation[2, 2] = cy * cx
    transformation[:3, 3] = pose[3:]
    return transformation
