"""Internal segmented tensor reductions."""

from __future__ import annotations

import torch


def segment_sum(
    values: torch.Tensor,
    ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Sum leading rows of ``values`` into segments selected by ``ids``.

    ``ids`` contains one segment index per leading row. Missing segments are
    retained as zero rows, and all trailing dimensions are reduced together.
    The operation uses ordinary PyTorch primitives and supports autograd with
    respect to ``values``.
    """
    if num_segments == 1:
        return values.sum(dim=0, keepdim=True)
    output = values.new_zeros((num_segments, *values.shape[1:]))
    output.index_add_(0, ids, values)
    return output


__all__ = ["segment_sum"]
