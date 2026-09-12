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


BACKENDS = ("torchpcl", "open3d-tensor", "open3d-legacy")


def make_case(prepared: PreparedCloud, backend: str, device: str = "cpu") -> Case:
    """Allocate inputs once and return an independently reusable case.

    Open3D exposes farthest-point downsampling twice: on the legacy
    ``geometry.PointCloud`` (CPU/float64 only) and on the tensor
    ``t.geometry.PointCloud`` (CPU and CUDA, honoring ``config.dtype``). They
    are separate backends because their runtimes differ by more than an order
    of magnitude, so neither one alone represents Open3D.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if backend == "open3d-legacy" and device != "cpu":
        raise ValueError("Open3D's legacy farthest_point_down_sample supports CPU only")
    config = prepared.config

    def synchronize():
        pass

    device_memory_reader = None
    compute_dtype = config.dtype

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
    elif backend == "open3d-tensor":
        import open3d as o3d
        import open3d.core as o3c

        if device == "cuda":
            synchronize = o3c.cuda.synchronize
            device_memory_reader = cuda_process_memory_reader()
        o3d_device = o3c.Device("CUDA:0" if device == "cuda" else "CPU:0")
        cloud = o3d.t.geometry.PointCloud(
            o3c.Tensor(prepared.points.astype(config.dtype), device=o3d_device))

        def run():
            return cloud.farthest_point_down_sample(config.num_samples)
    else:
        # Vector3dVector stores float64 whatever the configured input dtype is.
        import open3d as o3d

        compute_dtype = "float64"
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prepared.points))

        def run():
            return cloud.farthest_point_down_sample(config.num_samples)

    return Case(
        name="farthest_point_sample", backend=backend, device=device,
        run=run, synchronize=synchronize, device_memory_reader=device_memory_reader,
        metadata={"input_dtype": config.dtype, "compute_dtype": compute_dtype},
    )


__all__ = ["BACKENDS", "SamplingConfig", "PreparedCloud", "prepare_cloud", "make_case"]
