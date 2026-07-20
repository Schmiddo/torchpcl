"""Internal segmented tensor reductions."""

from __future__ import annotations

import torch


def segment_mean(
    values: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Average contiguous segments of leading rows delimited by ``offsets``.

    Repeated offsets retain empty segments as ``nan`` rows, and all trailing
    dimensions are reduced together. The operation supports autograd with
    respect to ``values``.
    """
    return torch.segment_reduce(values, "mean", offsets=offsets, axis=0)


def segment_sum(
    values: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Sum contiguous segments of leading rows delimited by ``offsets``.

    Repeated offsets retain empty segments as zero rows, and all trailing
    dimensions are reduced together. The operation supports autograd with
    respect to ``values``.
    """
    return torch.segment_reduce(values, "sum", offsets=offsets, axis=0)


__all__ = ["segment_mean", "segment_sum"]
