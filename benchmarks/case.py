"""A backend/operation-neutral benchmark case, reusable across benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


def select_devices(requested: str, cuda_available: bool, backend: str = "torchpcl",
                   reason: str | None = None):
    """Default to both devices; record unavailable CUDA, but reject explicit CUDA.

    `reason` overrides the explanation for a backend that has no CUDA support
    to lack, rather than one whose build happens not to have it.
    """
    if requested not in ("all", "cpu", "cuda"):
        raise ValueError("device must be all, cpu, or cuda")
    reason = reason or f"CUDA requires an accessible GPU and a CUDA {backend} build"
    if requested == "cuda" and not cuda_available:
        raise ValueError(reason)
    devices = ["cpu", "cuda"] if requested == "all" else [requested]
    skipped = []
    if "cuda" in devices and not cuda_available:
        devices.remove("cuda")
        skipped.append({"backend": backend, "device": "cuda", "reason": reason})
    return devices, skipped


def _no_metrics(result: Any) -> dict:
    return {}


@dataclass
class Case:
    """A prepared operation with explicit reset, synchronization, and evaluation hooks.

    ``run`` only invokes the timed operation; a runner can call ``run``
    directly and manage synchronization itself, or use
    ``run_synchronized``. Input preparation stays outside both. ``evaluate``
    turns a ``run()`` result into a dict of per-run numeric metrics (an
    empty dict if the operation has none, e.g. no ground truth to score
    against). ``metadata`` carries static descriptive fields that don't vary
    across runs (e.g. dtype, or whether a backend is a reference-only
    baseline), reported by a runner alongside the measured metrics.
    ``group`` names the input a case ran on when a suite measures several
    (a registration pair, a Redwood fragment pair), and joins the case's key.

    This shape is intentionally independent of any particular runner: an
    ASV benchmark method can call ``case.reset()``/``case.run()`` directly
    and let ASV manage its own timing, instead of going through
    ``benchmarks.runner.measure``.
    """

    name: str
    backend: str
    device: str
    run: Callable[[], Any]
    synchronize: Callable[[], None] = lambda: None
    reset: Callable[[], None] = lambda: None
    evaluate: Callable[[Any], dict] = _no_metrics
    device_memory_reader: Callable[[], int] | None = None
    metadata: dict = field(default_factory=dict)
    group: str | None = None

    def run_synchronized(self):
        self.synchronize()
        result = self.run()
        self.synchronize()
        return result


__all__ = ["Case", "select_devices"]
