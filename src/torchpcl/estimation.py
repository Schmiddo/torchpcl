"""Transformation estimation from gathered correspondence pairs.

All methods take already-gathered (K, 3) correspondence pairs in float64
and return a (4, 4) float64 delta transformation.
"""

import abc
from typing import ClassVar

import torch

from .transforms import pose_to_matrix


class TransformationEstimation(abc.ABC):
    requires_normals: ClassVar[bool] = False

    @abc.abstractmethod
    def compute_transformation(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_normals: torch.Tensor | None,
    ) -> torch.Tensor: ...


class PointToPoint(TransformationEstimation):
    """Closed-form point-to-point estimation (Umeyama / Kabsch)."""

    requires_normals: ClassVar[bool] = False

    def compute_transformation(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_mean = source.mean(dim=0)
        target_mean = target.mean(dim=0)
        h = (source - source_mean).T @ (target - target_mean)
        u, _, vh = torch.linalg.svd(h)
        v = vh.T
        d = torch.linalg.det(v @ u.T)
        s = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
        rotation = v @ s @ u.T

        transformation = torch.eye(4, dtype=source.dtype, device=source.device)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = target_mean - rotation @ source_mean
        return transformation


class PointToPlane(TransformationEstimation):
    """Linearized point-to-plane estimation (6x6 JtJ solve).

    Requires target normals.
    """

    requires_normals: ClassVar[bool] = True

    def compute_transformation(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_normals: torch.Tensor | None,
    ) -> torch.Tensor:
        if target_normals is None:
            raise ValueError("PointToPlane requires target normals")
        residuals = ((source - target) * target_normals).sum(dim=-1)
        jacobian = torch.cat(
            [torch.linalg.cross(source, target_normals), target_normals], dim=-1
        )
        jtj = jacobian.T @ jacobian
        jtr = jacobian.T @ residuals
        try:
            pose = torch.linalg.solve(jtj, -jtr)
        except torch.linalg.LinAlgError:
            pose = torch.linalg.lstsq(jtj, -jtr).solution
        return pose_to_matrix(pose)
