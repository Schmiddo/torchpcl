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


def fragment_points(path, dtype: str) -> np.ndarray:
    """Load one PLY fragment, quantized to `dtype` and stored float64.

    Real scans are surfaces: locally dense, with most of the bounding volume
    empty. Uniform points in a cube have neither property, and a spatial index
    is exactly the kind of structure whose ranking that difference can flip,
    so the operation benchmarks report both.
    """
    import open3d as o3d

    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: fragment not found; run `just download-data`")
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"{path}: expected at least three finite points")
    return points.astype(dtype).astype(np.float64)


__all__ = ["fragment_points", "random_points"]
