"""Shared, backend-neutral input generation for the operation benchmarks.

Kept separate from the per-operation `*_cases` modules so several of them can
build on the same clouds without depending on each other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def random_points(count: int, dtype: str, seed: int) -> np.ndarray:
    """Uniform points in the unit cube, quantized to `dtype`, stored float64.

    Quantizing before promoting back to float64 means every backend starts
    from identical coordinate values whatever precision it goes on to compute
    in -- the same quantize-then-promote contract the registration and
    sampling benchmarks use.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    rng = np.random.default_rng(seed)
    return rng.random((count, 3)).astype(dtype).astype(np.float64)


def load_fragment(path, dtype: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load points and optional normals once, quantized to dtype, stored float64.

    Real scans have locally dense surfaces unlike uniform random points. Use
    torchpcl's PLY loader so benchmark inputs follow the library's IO behavior.
    """
    import torch
    import torchpcl as tp

    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: fragment not found; run `just download-data`")
    cloud = tp.load_ply(path, dtype=getattr(torch, dtype))
    points = cloud.points.numpy().astype(np.float64)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"{path}: expected at least three finite points")
    normals = None if cloud.normals is None else cloud.normals.numpy().astype(np.float64)
    return points, normals


def random_normals(count: int, seed: int) -> np.ndarray:
    """Synthetic unit normals from a stream independent of random_points."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    normals = rng.normal(size=(count, 3))
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


__all__ = ["load_fragment", "random_points", "random_normals"]
