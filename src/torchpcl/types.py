"""Result and criteria types for the registration pipeline."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ICPConvergenceCriteria:
    """Convergence criteria for ICP.

    Iteration stops when the relative change of both fitness and inlier
    RMSE falls below the given thresholds, or after ``max_iteration``
    iterations.
    """

    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6
    max_iteration: int = 30


@dataclass
class RegistrationResult:
    """Result of a registration run or evaluation.

    Attributes:
        transformation: (4, 4) float64 source-to-target transformation.
        correspondences: (N,) int64 target index per source point,
            -1 where no correspondence was found within the search radius.
        fitness: Number of inlier correspondences divided by the number of
            source points. Higher is better.
        inlier_rmse: RMSE over inlier correspondences. Lower is better.
        converged: Whether the convergence criteria were met before
            ``max_iteration`` was reached.
        num_iterations: Number of ICP iterations executed.
    """

    transformation: torch.Tensor
    correspondences: torch.Tensor = field(repr=False)
    fitness: float = 0.0
    inlier_rmse: float = 0.0
    converged: bool = False
    num_iterations: int = 0
