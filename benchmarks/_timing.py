"""Shared timing helpers for torchpcl benchmarks."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import TypeVar, cast

import torch

T = TypeVar("T")
_MISSING = object()


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
) -> tuple[T, float]:
    """Warm up once and return the last result and median elapsed seconds.

    Device synchronization happens immediately outside each timed region. This
    measures completed CUDA work without charging earlier queued work to the
    operation.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")

    fn()
    synchronize(device)

    samples = []
    result: T | object = _MISSING
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)

    assert result is not _MISSING
    return cast(T, result), statistics.median(samples)


def torch_devices() -> list[torch.device]:
    """Return available torch devices as separate benchmark targets."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices
