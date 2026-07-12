"""Tensor-valued registration result types."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class RegistrationMetrics:
    """Metrics for a batch of source-to-target transforms."""

    transforms: torch.Tensor
    fitness: torch.Tensor
    inlier_rmse: torch.Tensor


@dataclass(frozen=True, eq=False)
class ICPResult(RegistrationMetrics):
    """Batched ICP output without materialized correspondences."""

    converged: torch.Tensor
    iterations: torch.Tensor


__all__ = ["ICPResult", "RegistrationMetrics"]
