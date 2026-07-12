"""Tensor-valued registration result types."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class ICPResult:
    """Batched ICP output without materialized correspondences."""

    transforms: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor
    fitness: torch.Tensor
    inlier_rmse: torch.Tensor


__all__ = ["ICPResult"]
