"""Argument parsing, environment reporting, and output helpers shared by the
command-line benchmark scripts. Reusing a `Case`/`measure` pair with a fresh
harness (an ASV adapter, a notebook) needs none of this module."""

from __future__ import annotations

import argparse
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value: str) -> float:
    import math

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def default_threads() -> int:
    """Prefer physical cores, falling back to the logical CPU count."""
    try:
        import psutil

        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def configure_threads(threads: int) -> None:
    """Set BLAS/OpenMP thread env vars. Call before importing Open3D/PyTorch."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(threads)


def require_open3d():
    """Import and return the `open3d` module, or exit with setup guidance."""
    try:
        import open3d as o3d
    except ImportError:
        raise SystemExit(
            "Open3D is required; see benchmarks/README.md for Python 3.12 setup"
        )
    return o3d


def environment_metadata(devices: list[str], *, open3d_version: str) -> dict:
    """Interpreter, platform, and library versions for the JSON report."""
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "open3d": open3d_version,
        "torchpcl": version("torchpcl"),
        "gpu": torch.cuda.get_device_name() if "cuda" in devices else None,
    }


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def format_optional(value: float | None, fmt: str = ".1f") -> str:
    """Render an optional measurement (e.g. a CPU-only case's VRAM) as ``-``."""
    return "-" if value is None else format(value, fmt)


__all__ = [
    "configure_threads",
    "default_threads",
    "environment_metadata",
    "format_optional",
    "positive_float",
    "positive_int",
    "require_open3d",
    "write_json",
]
