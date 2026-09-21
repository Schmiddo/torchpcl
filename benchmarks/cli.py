"""Argument parsing, environment reporting, and output helpers shared by the
command-line benchmark scripts. Reusing a `Case`/`measure` pair with a fresh
harness (an ASV adapter, a notebook) needs none of this module."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
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


#: small-gicp has no GPU implementation at all, which is a different situation
#: from a build that happens to lack CUDA.
SMALL_GICP_CPU_ONLY = "small-gicp is CPU only; it has no GPU implementation"


def require_small_gicp():
    """Import and return the `small_gicp` module, or exit with setup guidance."""
    try:
        import small_gicp
    except ImportError:
        raise SystemExit(
            "small-gicp is required; install the baseline dependency group "
            "(see benchmarks/README.md)"
        )
    return small_gicp


def source_revision() -> str | None:
    """``git describe``-style revision of the working tree, ``None`` outside git.

    Suffixed with ``-dirty`` when tracked files differ from the commit, since a
    report measuring uncommitted kernel edits cannot be reproduced from the
    hash alone.
    """
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    revision = head.stdout.strip()
    return revision + "-dirty" if dirty.stdout.strip() else revision


#: Environment variable overriding the derived machine label, for the case the
#: derivation cannot handle: two boxes with the same processor.
MACHINE_ENV = "TORCHPCL_BENCHMARK_MACHINE"


#: Trademark noise and clock speed in a /proc model name: neither identifies
#: the hardware, and both make the label unreadable.
CPU_NOISE = re.compile(r"\(R\)|\(TM\)|\bCPU\b|\bProcessor\b|@.*$", re.IGNORECASE)


def slug(text: str) -> str:
    """Lowercase, dash-joined, safe as both a filename and a chart label."""
    kept = "".join(c if c.isalnum() else "-" for c in text.lower())
    return "-".join(part for part in kept.split("-") if part)


def processor_model() -> str | None:
    """The CPU's marketing name, e.g. ``AMD Ryzen 9 7950X``.

    `platform.processor()` merely repeats the architecture on Linux, so read
    the model out of /proc; everything else falls back to the architecture.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("model name", "Model"):
                return CPU_NOISE.sub(" ", value).strip()
    except OSError:
        pass
    return platform.processor() or None


def machine_label() -> str:
    """A stable name for the computer that measured this, derived from its
    hardware rather than its hostname.

    Timings compare only within one computer, so a history still has to know
    which one -- but a hostname is a name that ends up committed and shared,
    and it says nothing about *why* two numbers are comparable. The processor
    does. Two boxes with the same processor collide; set ``$%s`` to tell them
    apart. Deliberately independent of PyTorch and of the selected devices, so
    the CPU-only baseline environment labels a machine exactly as the CUDA one
    does and their points share a card.
    """ % MACHINE_ENV
    override = os.environ.get(MACHINE_ENV, "").strip()
    if override:
        return slug(override)
    parts = [platform.system(), processor_model() or platform.machine()]
    return slug("-".join(part for part in parts if part)) or "unknown"


def environment_metadata(devices: list[str], *, open3d_version: str | None = None,
                         small_gicp_version: str | None = None) -> dict:
    """Interpreter, platform, library versions, and run identity for the report.

    ``timestamp``, ``revision`` and ``command`` are what let two saved reports be
    ordered and attributed; `compare.py` prints them when diffing.
    """
    import torch

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "revision": source_revision(),
        "command": sys.argv,
        # Which computer measured this. Timings are only comparable within one
        # machine, so a saved history keys its series on this; `compare.py`
        # has always printed the field. It is a hardware description, never the
        # hostname: the histories are committed, and a machine's name is not
        # something a repository should publish.
        "machine": machine_label(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "open3d": open3d_version,
        "small_gicp": small_gicp_version,
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
    "machine_label",
    "positive_float",
    "positive_int",
    "require_open3d",
    "require_small_gicp",
    "SMALL_GICP_CPU_ONLY",
    "source_revision",
    "write_json",
]
