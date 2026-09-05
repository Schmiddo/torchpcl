"""Prepare a shared random cloud and construct farthest-point-sample cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .case import Case
from .memory import cuda_process_memory_reader


@dataclass(frozen=True)
class SamplingConfig:
    num_points: int = 200_000
    num_samples: int = 5_000
    dtype: str = "float32"
    seed: int = 0

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.num_points, int) or self.num_points <= 0:
            raise ValueError("num_points must be a positive integer")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if self.num_samples >= self.num_points:
            raise ValueError("num_samples must be less than num_points")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")


@dataclass
class PreparedCloud:
    config: SamplingConfig
    points: np.ndarray

    def metadata(self):
        return {"num_points": self.points.shape[0]}


def prepare_cloud(config: SamplingConfig) -> PreparedCloud:
    """Generate a random cloud once; no device allocation is performed.

    Points are quantized to ``config.dtype`` before being stored as float64,
    so torchpcl and Open3D cases built from the same ``PreparedCloud`` start
    from identical coordinate values (mirrors
    ``registration_cases.prepare_pair``'s quantize-then-promote step).
    """
    rng = np.random.default_rng(config.seed)
    points = rng.random((config.num_points, 3)).astype(config.dtype).astype(np.float64)
    return PreparedCloud(config, points)


def make_case(prepared: PreparedCloud, backend: str, device: str = "cpu") -> Case:
    """Allocate inputs once and return an independently reusable case."""
    if backend not in ("torchpcl", "open3d"):
        raise ValueError("backend must be torchpcl or open3d")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if backend == "open3d" and device != "cpu":
        raise ValueError("Open3D farthest_point_down_sample supports CPU only")
    config = prepared.config

    def synchronize():
        pass

    device_memory_reader = None

    if backend == "torchpcl":
        import torch
        import torchpcl as tp

        if device == "cuda":
            synchronize = torch.cuda.synchronize
            device_memory_reader = cuda_process_memory_reader()
        dtype = getattr(torch, config.dtype)
        points = torch.tensor(prepared.points, dtype=dtype, device=device)

        def run():
            return tp.farthest_point_sample(points, config.num_samples)
    else:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prepared.points))

        def run():
            return cloud.farthest_point_down_sample(config.num_samples)

    return Case(
        name="farthest_point_sample", backend=backend, device=device,
        run=run, synchronize=synchronize, device_memory_reader=device_memory_reader,
        metadata={
            "input_dtype": config.dtype,
            "compute_dtype": config.dtype if backend == "torchpcl" else "float64",
            "reference_only": backend == "open3d",
        },
    )


__all__ = ["SamplingConfig", "PreparedCloud", "prepare_cloud", "make_case"]
