"""Prepare shared inputs and construct registration calls independently of a runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .case import Case, select_devices
from .metrics import load_transform, pose_errors
from .memory import cuda_process_memory_reader


@dataclass(frozen=True)
class RegistrationConfig:
    voxel_size: float = 0.05
    dtype: str = "float32"
    seed: int = 0
    icp_distance: float | None = None
    fgr_distance: float | None = None
    icp_iterations: int = 50
    fgr_iterations: int = 64

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        for name, multiplier in (("icp_distance", 5), ("fgr_distance", 0.5)):
            value = getattr(self, name)
            value = multiplier * self.voxel_size if value is None else value
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("icp_iterations", "fgr_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")


# Preparation parameters, as multiples of the voxel size. They define the
# workload rather than reproducing another library's: preparation is untimed
# and both backends receive the same arrays, so these are a recorded choice.
# `FEATURE_NEIGHBORS` is torchpcl's maximum; Open3D's own examples use 100.
BACKENDS = ("torchpcl", "open3d", "small-gicp")
OBJECTIVES = ("point-to-point", "point-to-plane")

NORMAL_RADIUS = 2
NORMAL_NEIGHBORS = 30
FEATURE_RADIUS = 5
FEATURE_NEIGHBORS = 64


@dataclass
class PreparedPair:
    """Plain arrays, so a backend adapter builds its own objects from them.

    Every array is quantized to the configured dtype and held as float64, so
    both backends start from identical coordinate values whatever their own
    storage is.
    """

    directory: Path
    config: RegistrationConfig
    expected: np.ndarray
    originals: tuple[np.ndarray, np.ndarray]
    clouds: tuple[np.ndarray, np.ndarray]
    normals: tuple[np.ndarray, np.ndarray] | None
    features: tuple[np.ndarray, np.ndarray] | None

    def metadata(self):
        return {
            "directory": str(self.directory.resolve()),
            "original_points": [len(points) for points in self.originals],
            "registration_points": [len(points) for points in self.clouds],
            "ground_truth": self.expected.tolist(),
            "preprocessing": {
                "provider": "torchpcl",
                "voxel_size": self.config.voxel_size,
                "normal_radius": NORMAL_RADIUS * self.config.voxel_size,
                "normal_neighbors": NORMAL_NEIGHBORS,
                **({} if self.features is None else {
                    "feature_radius": FEATURE_RADIUS * self.config.voxel_size,
                    "feature_neighbors": FEATURE_NEIGHBORS}),
            },
        }


def prepare_pair(directory: str | Path, config: RegistrationConfig, *,
                 with_normals=False, with_features=True):
    """Load and preprocess once; no device allocation or registration is performed."""
    directory = Path(directory)
    expected = load_transform(directory / "T_target_source.txt")
    originals, clouds, normals, features = [], [], [], []
    for name in ("source", "target"):
        path = directory / f"{name}.ply"
        original, cloud, normal, feature = prepare_cloud(
            path, config, with_normals=with_normals, with_features=with_features)
        originals.append(original)
        clouds.append(cloud)
        normals.append(normal)
        features.append(feature)
    return PreparedPair(directory, config, expected, tuple(originals), tuple(clouds),
                        tuple(normals) if with_normals or with_features else None,
                        tuple(features) if with_features else None)


def prepare_cloud(path: Path, config: RegistrationConfig, *, with_normals=False,
                  with_features=True):
    """Load one cloud and compute the shared, dtype-quantized benchmark inputs.

    torchpcl does the preprocessing, so a registration run needs no Open3D.
    What it produces is the solver's input, not a measurement: it sits outside
    every timed region and both backends receive it unchanged.
    """
    import torch
    import torchpcl as tp

    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = tp.load_ply(path, dtype=torch.float64)
    original = loaded.points.numpy()
    if len(original) < 3 or not np.isfinite(original).all():
        raise ValueError(f"{path}: expected at least three finite points")
    centroids = tp.voxelize(loaded.points, config.voxel_size).reduce(
        loaded.points, "mean")
    if len(centroids) < 10:
        raise ValueError(f"{path}: fewer than 10 downsampled points; reduce voxel_size")
    # Quantize before promoting back to float64 storage, so a float32 run and a
    # float64 run of the same configuration differ only where they should.
    points = _quantize(centroids.numpy(), config)
    if not (with_normals or with_features):
        return original, points, None, None

    dtype = getattr(torch, config.dtype)
    quantized = torch.tensor(points, dtype=dtype)
    normals = tp.estimate_normals(
        quantized, NORMAL_RADIUS * config.voxel_size, k=NORMAL_NEIGHBORS).normals
    oriented = tp.orient_normals_toward_viewpoint(
        tp.PointCloud.from_points(quantized, normals=normals),
        torch.zeros(3, dtype=dtype))
    shared_normals = _quantize(oriented.normals.numpy(), config)
    if not with_features:
        return original, points, shared_normals, None
    features = tp.fpfh(oriented, FEATURE_RADIUS * config.voxel_size,
                       max_neighbors=FEATURE_NEIGHBORS)
    return (original, points, shared_normals,
            _quantize(features.descriptors.numpy(), config))


def _quantize(array: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    return array.astype(config.dtype).astype(np.float64)


def _open3d_cloud(o3d, points: np.ndarray):
    """A legacy Open3D cloud from shared points, built in the adapter."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def _open3d_feature(o3d, descriptors: np.ndarray):
    """Open3D stores FPFH transposed: 33 rows by N columns."""
    feature = o3d.pipelines.registration.Feature()
    feature.data = descriptors.T.copy()
    return feature


def _open3d_quality(pair: PreparedPair, distance: float, estimate: np.ndarray) -> dict:
    """Open3D's fitness and inlier RMSE, against the full-resolution clouds.

    Pose error is scored against the dataset's ground truth and needs no other
    implementation. These two do, so they are requested explicitly rather than
    computed on every run.
    """
    import open3d as o3d

    clouds = [_open3d_cloud(o3d, points) for points in pair.originals]
    metrics = o3d.pipelines.registration.evaluate_registration(
        *clouds, distance, estimate)
    return {"fitness": metrics.fitness, "inlier_rmse": metrics.inlier_rmse}


def make_case(pair: PreparedPair, method: str, backend: str, device: str = "cpu",
              *, open3d_metrics: bool = False, objective: str = "point-to-point",
              threads: int = 1) -> Case:
    """Allocate inputs/options once and return an independently reusable case."""
    if method not in ("icp", "fgr") or backend not in BACKENDS:
        raise ValueError(f"expected method icp/fgr and backend "
                         f"{'/'.join(BACKENDS)}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}")
    if backend == "open3d" and method == "fgr" and device != "cpu":
        raise ValueError("Open3D FGR supports CPU only")
    if backend == "small-gicp":
        if method == "fgr":
            raise ValueError("small-gicp has no global registration")
        if device != "cpu":
            raise ValueError("small-gicp is CPU only")
    if method == "fgr" and pair.features is None:
        raise ValueError("FGR requires prepare_pair(..., with_features=True)")
    if method == "icp" and objective == "point-to-plane" and pair.normals is None:
        raise ValueError("point-to-plane requires prepare_pair(..., with_normals=True)")
    config = pair.config
    distance = getattr(config, f"{method}_distance")
    iterations = getattr(config, f"{method}_iterations")

    def synchronize():
        pass

    device_memory_reader = None

    if backend == "torchpcl":
        import torch
        import torchpcl as tp
        from torchpcl import _C

        if device == "cuda":
            select_devices("cuda", torch.cuda.is_available() and _C.has_cuda())
            synchronize = torch.cuda.synchronize
            device_memory_reader = cuda_process_memory_reader()
        dtype = getattr(torch, config.dtype)
        clouds = [tp.PointCloud.from_points(torch.tensor(
            points, dtype=dtype, device=device)) for points in pair.clouds]
        if method == "icp":
            init = torch.eye(4, dtype=dtype, device=device)
            levels = [tp.ICPLevel(max_correspondence_distance=distance, max_iterations=iterations)]
            options = None
            if objective == "point-to-plane":
                # Only the target's normals are used, by the point-to-plane residual.
                clouds[1] = tp.PointCloud.from_points(
                    clouds[1].points,
                    normals=torch.tensor(pair.normals[1], dtype=dtype, device=device))
                options = tp.ICPOptions(objective=tp.PointToPlane())
            def run():
                return tp.icp(*clouds, levels, init=init, options=options)
        else:
            clouds = [cloud.with_features(torch.tensor(
                descriptors, dtype=dtype, device=device
            )) for cloud, descriptors in zip(clouds, pair.features)]
            options = tp.FGROptions(max_correspondence_distance=distance,
                                    max_iterations=iterations, seed=config.seed)
            def run():
                return tp.fast_global_registration(*clouds, options=options)
    elif backend == "small-gicp":
        import small_gicp

        # Both sides receive the same points. small-gicp's normals are its own:
        # its PointCloud has no normal setter, so point-to-plane compares two
        # implementations' normals rather than one shared array.
        target = small_gicp.PointCloud(np.ascontiguousarray(pair.clouds[1]))
        source = small_gicp.PointCloud(np.ascontiguousarray(pair.clouds[0]))
        tree = small_gicp.KdTree(target, num_threads=threads)
        kind = "ICP" if objective == "point-to-point" else "PLANE_ICP"
        if kind == "PLANE_ICP":
            small_gicp.estimate_normals(target, tree, num_neighbors=NORMAL_NEIGHBORS,
                                        num_threads=threads)
        def run():
            return small_gicp.align(
                target, source, tree, registration_type=kind,
                max_correspondence_distance=distance, num_threads=threads,
                max_iterations=iterations)
    else:
        import open3d as o3d
        registration = o3d.pipelines.registration
        if method == "icp":
            registration = o3d.t.pipelines.registration
            if device == "cuda":
                select_devices("cuda", o3d.core.cuda.is_available(), backend="open3d")
                synchronize = o3d.core.cuda.synchronize
                device_memory_reader = cuda_process_memory_reader()
            o3d_device = o3d.core.Device("CUDA:0" if device == "cuda" else "CPU:0")
            o3d_dtype = o3d.core.Dtype.Float32 if config.dtype == "float32" else o3d.core.Dtype.Float64
            clouds = [o3d.t.geometry.PointCloud(o3d.core.Tensor(
                points, dtype=o3d_dtype, device=o3d_device
            )) for points in pair.clouds]
            # Open3D's tensor ICP API keeps its pose in CPU float64, even for
            # float32 clouds on CUDA. Point storage/search use the selected dtype/device.
            init = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64, o3d.core.Device("CPU:0"))
            if objective == "point-to-plane":
                clouds[1].point.normals = o3d.core.Tensor(
                    pair.normals[1], dtype=o3d_dtype, device=o3d_device)
                estimation = registration.TransformationEstimationPointToPlane()
            else:
                estimation = registration.TransformationEstimationPointToPoint()
            criteria = registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iterations)
            def run():
                return registration.icp(*clouds, distance, init, estimation, criteria,
                                        voxel_size=-1.0)
        else:
            options = registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance, iteration_number=iterations,
                decrease_mu=True, division_factor=1.4, use_absolute_scale=False,
                tuple_scale=0.95, maximum_tuple_count=1000, tuple_test=True)
            clouds = [_open3d_cloud(o3d, points) for points in pair.clouds]
            features = [_open3d_feature(o3d, descriptors)
                        for descriptors in pair.features]
            def run():
                return registration.registration_fgr_based_on_feature_matching(
                    *clouds, *features, options)

    def reset():
        # Only Open3D carries global solver state; torchpcl seeds per call.
        if backend == "open3d":
            import open3d as o3d
            o3d.utility.random.seed(config.seed)

    def evaluate(result):
        if backend == "torchpcl":
            estimate = result.transforms[0].detach().cpu().numpy()
        elif backend == "small-gicp":
            estimate = np.asarray(result.T_target_source)
        elif method == "icp":
            estimate = result.transformation.cpu().numpy()
        else:
            estimate = np.asarray(result.transformation)
        if not np.isfinite(estimate).all():
            raise ValueError(f"{method}/{backend}/{device}: non-finite transform")
        metrics = {
            **pose_errors(estimate, pair.expected, pair.originals[0]),
            "transform": estimate.tolist(),
            "solver_success": bool(result.success.item())
            if backend == "torchpcl" and method == "fgr" else None,
        }
        if open3d_metrics:
            metrics.update(_open3d_quality(pair, distance, estimate))
        return metrics

    return Case(
        name=method, backend=backend, device=device,
        run=run, synchronize=synchronize, reset=reset, evaluate=evaluate,
        device_memory_reader=device_memory_reader,
        metadata={
            "input_dtype": config.dtype,
            "compute_dtype": (config.dtype if backend == "torchpcl" else
                              "float64" if backend == "small-gicp" else
                              "mixed" if method == "icp" and config.dtype == "float32"
                              else "float64"),
            "objective": objective if method == "icp" else None,
            "implementation": "tensor" if backend == "open3d" and method == "icp" else
                              "legacy" if backend == "open3d" else backend,
            "transform_dtype": ("float64" if backend in ("open3d", "small-gicp")
                                else config.dtype),
        },
    )
