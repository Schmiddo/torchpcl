"""Prepare a shared cloud with normals and construct FPFH cases per backend.

Both libraries compute 33-bin histograms over bounded hybrid neighborhoods
(a radius capped at a neighbor count), which is the mode Open3D recommends on
GPU and the only mode torchpcl offers. Normals are estimated once during
preparation and handed to both backends, so only the descriptor computation
differs -- the same shared-preprocessing contract the registration benchmark
uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .case import Case
from .memory import cuda_process_memory_reader

OPS = ("fpfh",)
BACKENDS = ("torchpcl", "open3d")
# torchpcl's FPFH accepts at most this many neighbors per point.
MAX_NEIGHBORS = 64
# Descriptors sum to 600 per row, so this is a ~0.17% disagreement: well above
# float32 noise, well below a genuinely different neighborhood. A max over
# 200k x 33 values is dominated by single differing neighborhoods, so the gate
# counts affected rows instead -- Open3D's own CPU and CUDA results differ by
# several units under a max.
BIN_TOLERANCE = 1.0


@dataclass(frozen=True)
class FeaturesConfig:
    radius: float = 0.03
    max_neighbors: int = MAX_NEIGHBORS
    normal_k: int = 30
    dtype: str = "float32"

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        if (not isinstance(self.max_neighbors, int)
                or not 2 <= self.max_neighbors <= MAX_NEIGHBORS):
            raise ValueError(f"max_neighbors must be in [2, {MAX_NEIGHBORS}]")
        if not isinstance(self.normal_k, int) or self.normal_k <= 0:
            raise ValueError("normal_k must be a positive integer")


@dataclass
class PreparedFeatures:
    config: FeaturesConfig
    source: str
    points: np.ndarray
    normals: np.ndarray

    def metadata(self):
        return {"source": self.source, "num_points": int(self.points.shape[0]),
                "normal_k": self.config.normal_k}


def prepare_features(config: FeaturesConfig, points: np.ndarray,
                     source: str = "random") -> PreparedFeatures:
    """Estimate normals once; both backends receive this exact array."""
    import open3d as o3d
    import open3d.core as o3c

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    cloud = o3d.t.geometry.PointCloud(o3c.Tensor(points.astype(config.dtype)))
    cloud.estimate_normals(max_nn=config.normal_k)
    cloud.orient_normals_towards_camera_location(o3c.Tensor(
        np.zeros(3, dtype=config.dtype)))
    return PreparedFeatures(
        config=config, source=source, points=points,
        normals=cloud.point.normals.numpy().astype(np.float64))


def _torchpcl_run(prepared, device):
    import torch
    import torchpcl as tp

    config = prepared.config
    dtype = getattr(torch, config.dtype)
    points = torch.tensor(prepared.points, dtype=dtype, device=device)
    cloud = tp.PointCloud(
        points,
        torch.tensor([0, len(points)], dtype=torch.int64, device=device),
        torch.tensor(prepared.normals, dtype=dtype, device=device))
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


def reference_output(prepared: PreparedFeatures):
    """Open3D's CPU descriptors, the target both backends are scored on."""
    return _open3d_run(prepared, "cpu")().numpy().astype(np.float64)


def _make_evaluate(prepared, backend, reference):
    if reference is None:
        return None

    def evaluate(result):
        if backend == "torchpcl":
            valid = result.valid.cpu().numpy()
            found = result.descriptors.cpu().numpy().astype(np.float64)
            # Surfaced rather than hidden: a filled neighbor list means the
            # bounded search may have dropped radius neighbors, so the two
            # libraries could be summarizing different neighborhoods.
            extra = {
                "valid_fraction": float(valid.mean()),
                "limit_reached_fraction": float(
                    result.neighbor_limit_reached.cpu().numpy().mean()),
            }
        else:
            found = result.cpu().numpy().astype(np.float64)
            valid = np.ones(len(found), dtype=bool)
            extra = {}
        if not valid.any():
            return {"descriptor_mismatch_fraction": 0.0, "max_bin_error": 0.0,
                    **extra}
        difference = np.abs(found[valid] - reference[valid])
        return {
            "descriptor_mismatch_fraction": float(
                np.mean(difference.max(1) > BIN_TOLERANCE)),
            "max_bin_error": float(difference.max()),
            **extra,
        }

    return evaluate


def make_case(prepared: PreparedFeatures, op: str, backend: str,
              device: str = "cpu", reference=None) -> Case:
    """Allocate inputs once and return an independently reusable case."""
    if op not in OPS:
        raise ValueError(f"op must be one of {', '.join(OPS)}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")

    def synchronize():
        pass

    device_memory_reader = None
    if device == "cuda":
        device_memory_reader = cuda_process_memory_reader()

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

    evaluate = _make_evaluate(prepared, backend, reference)
    return Case(
        name=op, backend=backend, device=device,
        run=run, synchronize=synchronize,
        device_memory_reader=device_memory_reader,
        **({} if evaluate is None else {"evaluate": evaluate}),
        metadata={"input_dtype": prepared.config.dtype},
    )


__all__ = [
    "BACKENDS", "MAX_NEIGHBORS", "OPS", "FeaturesConfig", "PreparedFeatures",
    "make_case", "prepare_features", "reference_output",
]
