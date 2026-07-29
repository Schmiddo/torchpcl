"""Shared timing helpers for torchpcl benchmarks."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import torch

T = TypeVar("T")
_MISSING = object()


@dataclass(frozen=True)
class TimingResult:
    """Measured samples and compact summary statistics."""

    samples_seconds: tuple[float, ...]
    median_seconds: float
    iqr_seconds: float


def synchronize(device: torch.device | str | None) -> None:
    """Synchronize a CUDA device; CPU and ``None`` are no-ops."""
    if device is None:
        return
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def timed(
    fn: Callable[[], T],
    repeats: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[T, TimingResult]:
    """Warm up five times and return the last result plus measured timing samples.

    Device synchronization happens immediately outside each timed region. This
    measures completed CUDA work without charging earlier queued work to the
    operation.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")

    for _ in range(5):
        fn()
    synchronize(device)

    samples: list[float] = []
    result: T | object = _MISSING
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)

    assert result is not _MISSING
    if len(samples) == 1:
        iqr = 0.0
    else:
        lower, _, upper = statistics.quantiles(
            samples, n=4, method="inclusive"
        )
        iqr = upper - lower
    return cast(T, result), TimingResult(
        samples_seconds=tuple(samples),
        median_seconds=statistics.median(samples),
        iqr_seconds=iqr,
    )


__all__ = ["TimingResult", "synchronize", "timed"]
