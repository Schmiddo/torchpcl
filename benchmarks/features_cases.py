"""FPFH cases using identical points and normals with bounded neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .case import Case
from .memory import cuda_process_memory_reader

OPS = ("fpfh",)
BACKENDS = ("torchpcl", "open3d")
# small-gicp computes no features; it is a registration and search library.
BACKEND_OPS = {"torchpcl": OPS, "open3d": OPS, "small-gicp": ()}
# torchpcl's FPFH accepts at most this many neighbors per point.
MAX_NEIGHBORS = 64


@dataclass(frozen=True)
class FeaturesConfig:
    radius: float = 0.03
    max_neighbors: int = MAX_NEIGHBORS
    dtype: str = "float32"

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        if (not isinstance(self.max_neighbors, int)
                or not 2 <= self.max_neighbors <= MAX_NEIGHBORS):
            raise ValueError(f"max_neighbors must be in [2, {MAX_NEIGHBORS}]")


@dataclass
class PreparedFeatures:
    config: FeaturesConfig
    source: str
    points: np.ndarray
    normals: np.ndarray

    def metadata(self):
        return {"source": self.source, "num_points": int(self.points.shape[0])}


def prepare_features(config: FeaturesConfig, points: np.ndarray,
                     normals: np.ndarray, source: str = "random") -> PreparedFeatures:
    """Validate and quantize the shared input normals before backend setup."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    normals = np.asarray(normals, dtype=config.dtype).astype(np.float64)
    if (normals.shape != points.shape or not np.isfinite(normals).all()
            or not np.any(normals != 0, axis=1).all()):
        raise ValueError("FPFH normals must be finite, nonzero, and match points' (N, 3) shape")
    return PreparedFeatures(config, source, points, normals)


def _torchpcl_run(prepared, device):
    import torch
    import torchpcl as tp

    config = prepared.config
    dtype = getattr(torch, config.dtype)
    points = torch.tensor(prepared.points, dtype=dtype, device=device)
    cloud = tp.PointCloud.from_points(
        points, normals=torch.tensor(prepared.normals, dtype=dtype, device=device))
    return lambda: tp.fpfh(cloud, config.radius,
                          max_neighbors=config.max_neighbors)


def _open3d_run(prepared, device):
    import open3d as o3d
    import open3d.core as o3c

    config = prepared.config
    dtype = o3c.float32 if config.dtype == "float32" else o3c.float64
    o3d_device = o3c.Device("CUDA:0" if device == "cuda" else "CPU:0")
    cloud = o3d.t.geometry.PointCloud(
        o3c.Tensor(prepared.points.astype(config.dtype), dtype, o3d_device))
    cloud.point.normals = o3c.Tensor(
        prepared.normals.astype(config.dtype), dtype, o3d_device)
    return lambda: o3d.t.pipelines.registration.compute_fpfh_feature(
        cloud, max_nn=config.max_neighbors, radius=config.radius)


def make_case(prepared: PreparedFeatures, op: str, backend: str,
              device: str = "cpu") -> Case:
    """Allocate inputs once and return an independently reusable case."""
    if op not in OPS:
        raise ValueError(f"op must be one of {', '.join(OPS)}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")

    if backend == "torchpcl":
        import torch

        run = _torchpcl_run(prepared, device)
        if device == "cuda":
            synchronize = torch.cuda.synchronize
    else:
        import open3d.core as o3c

        run = _open3d_run(prepared, device)
        if device == "cuda":
            synchronize = o3c.cuda.synchronize

    case = Case(name=op, backend=backend, device=device, run=run,
                metadata={"input_dtype": prepared.config.dtype})
    if device == "cuda":
        case.synchronize = synchronize
        case.device_memory_reader = cuda_process_memory_reader()
    return case


__all__ = [
    "BACKENDS", "BACKEND_OPS", "MAX_NEIGHBORS", "OPS", "FeaturesConfig",
    "PreparedFeatures",
    "make_case", "prepare_features",
]
