"""Transformation estimation from gathered correspondence pairs.

All methods take already-gathered (K, 3) correspondence pairs (float32 or
float64) and return a (4, 4) float64 delta transformation. The large
per-point reductions run on the input device in the input dtype; the
small final solves always run on the host in float64.
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
        # The 3x3 SVD costs more in cuSOLVER dispatch than in math; solve
        # it on the host in float64 (single stacked transfer of 15 values).
        small = torch.cat([h.reshape(9), source_mean, target_mean]).cpu().to(torch.float64)
        h_c, source_mean_c, target_mean_c = small[:9].reshape(3, 3), small[9:12], small[12:]
        u, _, vh = torch.linalg.svd(h_c)
        v = vh.T
        d = torch.linalg.det(v @ u.T)
        s = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
        rotation = v @ s @ u.T

        transformation = torch.eye(4, dtype=torch.float64)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = target_mean_c - rotation @ source_mean_c
        return transformation.to(source.device)


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
        # Solve the 6x6 on the host in float64 (single stacked transfer of
        # 42 values); pose_to_matrix on a CPU pose builds the 4x4 cheaply.
        small = torch.cat([jtj.reshape(36), jtr]).cpu().to(torch.float64)
        jtj_c, jtr_c = small[:36].reshape(6, 6), small[36:]
        try:
            pose = torch.linalg.solve(jtj_c, -jtr_c)
        except torch.linalg.LinAlgError:
            pose = torch.linalg.lstsq(jtj_c, -jtr_c).solution
        return pose_to_matrix(pose).to(source.device)
