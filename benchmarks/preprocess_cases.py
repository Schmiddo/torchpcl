"""Prepare a shared cloud and construct preprocessing cases per backend.

Parameter conventions were established by measurement, not by reading docs
(see `plans/benchmark-ops.md`):

- Both outlier filters map their neighbor count **directly**: torchpcl's `k`
  equals Open3D's `nb_neighbors` / `nb_points`, reproducing its masks exactly.
- torchpcl's `voxelize` returns a `Voxelization`, not a cloud. Only
  `voxelize` plus `reduce(points, "mean")` is like-for-like with Open3D's
  `voxel_down_sample`, and it reproduces it bit for bit. Both are timed, so
  the cost of the reduction is visible rather than quietly omitted.
- Normals are compared k-NN to k-NN. Open3D's optional `radius` changes its
  result, so it is not passed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .case import Case
from .memory import cuda_process_memory_reader

OPS = ("voxelize", "voxelize-reduce", "normals",
       "statistical-outliers", "radius-outliers")
BACKENDS = ("torchpcl", "open3d")


@dataclass(frozen=True)
class PreprocessConfig:
    voxel_size: float = 0.02
    normal_k: int = 30
    outlier_k: int = 20
    std_ratio: float = 2.0
    outlier_radius: float = 0.02
    dtype: str = "float32"

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        for name in ("voxel_size", "std_ratio", "outlier_radius"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("normal_k", "outlier_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class PreparedCloud:
    config: PreprocessConfig
    source: str
    points: np.ndarray

    def metadata(self):
        return {"source": self.source, "num_points": int(self.points.shape[0])}


def prepare_cloud(config: PreprocessConfig, points: np.ndarray,
                  source: str = "random") -> PreparedCloud:
    """Hold the shared cloud; every case quantizes from this one array."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return PreparedCloud(config=config, source=source, points=points)


def _torchpcl_run(prepared, op, device):
    import torch
    import torchpcl as tp

    config = prepared.config
    dtype = getattr(torch, config.dtype)
    points = torch.tensor(prepared.points, dtype=dtype, device=device)

    if op == "voxelize":
        return lambda: tp.voxelize(points, config.voxel_size)
    if op == "voxelize-reduce":
        def voxelize_reduce():
            return tp.voxelize(points, config.voxel_size).reduce(points, "mean")

        return voxelize_reduce
    if op == "normals":
        return lambda: tp.estimate_normals(points, k=config.normal_k)
    if op == "statistical-outliers":
        return lambda: tp.statistical_outliers(
            points, config.outlier_k, std_ratio=config.std_ratio)
    return lambda: tp.radius_outliers(
        points, config.outlier_radius, config.outlier_k)


def _open3d_run(prepared, op, device):
    import open3d as o3d
    import open3d.core as o3c

    config = prepared.config
    dtype = o3c.float32 if config.dtype == "float32" else o3c.float64
    o3d_device = o3c.Device("CUDA:0" if device == "cuda" else "CPU:0")
    cloud = o3d.t.geometry.PointCloud(
        o3c.Tensor(prepared.points.astype(config.dtype), dtype, o3d_device))

    if op in ("voxelize", "voxelize-reduce"):
        # Open3D has no split between the two; the same call answers both, and
        # comparing it against bare `voxelize` is what overcharges torchpcl.
        return lambda: cloud.voxel_down_sample(config.voxel_size)
    if op == "normals":
        def normals():
            estimated = cloud.clone()
            estimated.estimate_normals(max_nn=config.normal_k)
            return estimated

        return normals
    if op == "statistical-outliers":
        return lambda: cloud.remove_statistical_outliers(
            nb_neighbors=config.outlier_k, std_ratio=config.std_ratio)
    return lambda: cloud.remove_radius_outliers(
        nb_points=config.outlier_k, search_radius=config.outlier_radius)


# Far above float32 centroid noise (order 1e-7) and far below any useful
# voxel size, so it identifies voxels without merging neighbors.
CENTROID_TOLERANCE = 1e-5


def _centroid_keys(points: np.ndarray) -> set:
    """Tolerance-quantized centroid identities, comparable as a set.

    Comparing sorted rows would make a single differing voxel shift every row
    after it, reporting a one-voxel disagreement as a large coordinate error.
    Set membership reports it as the small disagreement it is -- which matters
    here, because Open3D's own CPU and CUDA voxelizations do differ on a
    handful of voxels.
    """
    return set(map(tuple, np.round(points / CENTROID_TOLERANCE).astype(np.int64)))


def _make_evaluate(prepared, op, backend, reference):
    """Score against the reference backend's output, or nothing to score."""
    if reference is None:
        return None

    if op in ("voxelize", "voxelize-reduce"):
        if op == "voxelize":
            # Bare voxelize returns no centroids, so there is nothing to
            # compare; its companion case carries the accuracy check.
            return None

        def evaluate(result):
            found = (result.cpu().numpy() if backend == "torchpcl"
                     else result.point.positions.cpu().numpy())
            keys = _centroid_keys(found.astype(np.float64))
            return {"centroid_mismatch_fraction":
                    len(reference - keys) / len(reference)}

        return evaluate

    if op == "normals":
        def evaluate(result):
            found = (result.normals.cpu().numpy() if backend == "torchpcl"
                     else result.point.normals.cpu().numpy())
            # torchpcl's normal signs are arbitrary without attached normals,
            # so only the unsigned direction is contractual.
            alignment = np.abs((found.astype(np.float64) * reference).sum(1))
            return {"min_normal_alignment": float(alignment.min())}

        return evaluate

    def evaluate(result):
        if backend == "torchpcl":
            kept = ~result.cpu().numpy()
        else:
            kept = result[1].cpu().numpy().astype(bool)
        return {"mask_mismatch_fraction": float(np.mean(kept != reference))}

    return evaluate


def reference_output(prepared: PreparedCloud, op: str):
    """Compute Open3D's CPU answer once, as the target every case is scored on.

    Open3D is the reference because it is the established implementation; a
    case scoring zero reproduces it exactly.
    """
    if op == "voxelize":
        return None
    result = _open3d_run(prepared, op, "cpu")()
    if op == "voxelize-reduce":
        return _centroid_keys(
            result.point.positions.numpy().astype(np.float64))
    if op == "normals":
        return result.point.normals.numpy().astype(np.float64)
    return result[1].numpy().astype(bool)


def make_case(prepared: PreparedCloud, op: str, backend: str,
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

        run = _torchpcl_run(prepared, op, device)
        if device == "cuda":
            synchronize = torch.cuda.synchronize
    else:
        import open3d.core as o3c

        run = _open3d_run(prepared, op, device)
        if device == "cuda":
            synchronize = o3c.cuda.synchronize

    evaluate = _make_evaluate(prepared, op, backend, reference)
    return Case(
        name=op, backend=backend, device=device,
        run=run, synchronize=synchronize,
        device_memory_reader=device_memory_reader,
        **({} if evaluate is None else {"evaluate": evaluate}),
        metadata={"input_dtype": prepared.config.dtype},
    )


__all__ = [
    "BACKENDS", "OPS", "PreparedCloud", "PreprocessConfig", "make_case",
    "prepare_cloud", "reference_output",
]
