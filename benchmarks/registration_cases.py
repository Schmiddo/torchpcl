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


@dataclass
class PreparedPair:
    directory: Path
    config: RegistrationConfig
    expected: np.ndarray
    originals: tuple[Any, Any]
    clouds: tuple[Any, Any]
    features: tuple[Any, Any] | None

    def metadata(self):
        return {
            "directory": str(self.directory.resolve()),
            "original_points": [len(c.points) for c in self.originals],
            "registration_points": [len(c.points) for c in self.clouds],
            "ground_truth": self.expected.tolist(),
        }


def prepare_pair(directory: str | Path, config: RegistrationConfig, *, with_features=True):
    """Load and preprocess once; no device allocation or registration is performed."""
    directory = Path(directory)
    expected = load_transform(directory / "T_target_source.txt")
    originals, clouds, features = [], [], []
    for name in ("source", "target"):
        path = directory / f"{name}.ply"
        original, cloud, feature = prepare_cloud(path, config, with_features=with_features)
        originals.append(original)
        clouds.append(cloud)
        if with_features:
            features.append(feature)
    return PreparedPair(directory, config, expected, tuple(originals), tuple(clouds),
                        tuple(features) if with_features else None)


def prepare_cloud(path: Path, config: RegistrationConfig, *, with_features=True):
    """Load one cloud and compute the shared, dtype-quantized benchmark inputs."""
    import open3d as o3d

    feature = None
    if not path.is_file():
        raise FileNotFoundError(path)
    original = o3d.io.read_point_cloud(str(path))
    points = np.asarray(original.points)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"{path}: expected at least three finite points")
    cloud = original.voxel_down_sample(config.voxel_size)
    if len(cloud.points) < 10:
        raise ValueError(f"{path}: fewer than 10 downsampled points; reduce voxel_size")
    # Quantize shared inputs before promoting them to legacy Open3D's double
    # storage. Both backends thus start from identical coordinate values.
    cloud.points = o3d.utility.Vector3dVector(np.asarray(cloud.points).astype(config.dtype))
    if with_features:
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=2 * config.voxel_size, max_nn=30))
        cloud.orient_normals_towards_camera_location(np.zeros(3))
        feature = o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(
                radius=5 * config.voxel_size, max_nn=100))
        feature.data = np.asarray(feature.data).astype(config.dtype).astype(np.float64)
    return original, cloud, feature


def make_case(pair: PreparedPair, method: str, backend: str, device: str = "cpu") -> Case:
    """Allocate inputs/options once and return an independently reusable case."""
    if method not in ("icp", "fgr") or backend not in ("torchpcl", "open3d"):
        raise ValueError("expected method icp/fgr and backend torchpcl/open3d")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if backend == "open3d" and method == "fgr" and device != "cpu":
        raise ValueError("Open3D FGR supports CPU only")
    if method == "fgr" and pair.features is None:
        raise ValueError("FGR requires prepare_pair(..., with_features=True)")
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
            np.asarray(cloud.points), dtype=dtype, device=device)) for cloud in pair.clouds]
        if method == "icp":
            init = torch.eye(4, dtype=dtype, device=device)
            levels = [tp.ICPLevel(max_correspondence_distance=distance, max_iterations=iterations)]
            def run():
                return tp.icp(*clouds, levels, init=init)
        else:
            clouds = [cloud.with_features(torch.tensor(
                np.asarray(feature.data).T, dtype=dtype, device=device
            )) for cloud, feature in zip(clouds, pair.features)]
            options = tp.FGROptions(max_correspondence_distance=distance,
                                    max_iterations=iterations, seed=config.seed)
            def run():
                return tp.fast_global_registration(*clouds, options=options)
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
                np.asarray(cloud.points), dtype=o3d_dtype, device=o3d_device
            )) for cloud in pair.clouds]
            # Open3D's tensor ICP API keeps its pose in CPU float64, even for
            # float32 clouds on CUDA. Point storage/search use the selected dtype/device.
            init = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64, o3d.core.Device("CPU:0"))
            objective = registration.TransformationEstimationPointToPoint()
            criteria = registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iterations)
            def run():
                return registration.icp(*clouds, distance, init, objective, criteria, voxel_size=-1.0)
        else:
            options = registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance, iteration_number=iterations,
                decrease_mu=True, division_factor=1.4, use_absolute_scale=False,
                tuple_scale=0.95, maximum_tuple_count=1000, tuple_test=True)
            def run():
                return registration.registration_fgr_based_on_feature_matching(
                    *pair.clouds, *pair.features, options)

    def reset():
        import open3d as o3d
        o3d.utility.random.seed(config.seed)

    def evaluate(result):
        import open3d as o3d

        if backend == "torchpcl":
            estimate = result.transforms[0].detach().cpu().numpy()
        elif method == "icp":
            estimate = result.transformation.cpu().numpy()
        else:
            estimate = np.asarray(result.transformation)
        if not np.isfinite(estimate).all():
            raise ValueError(f"{method}/{backend}/{device}: non-finite transform")
        metrics = o3d.pipelines.registration.evaluate_registration(
            *pair.originals, distance, estimate)
        return {
            **pose_errors(estimate, pair.expected, np.asarray(pair.originals[0].points)),
            "fitness": metrics.fitness, "inlier_rmse": metrics.inlier_rmse,
            "transform": estimate.tolist(),
            "solver_success": bool(result.success.item())
            if backend == "torchpcl" and method == "fgr" else None,
        }

    return Case(
        name=method, backend=backend, device=device,
        run=run, synchronize=synchronize, reset=reset, evaluate=evaluate,
        device_memory_reader=device_memory_reader,
        metadata={
            "input_dtype": config.dtype,
            "compute_dtype": (config.dtype if backend == "torchpcl" else
                              "mixed" if method == "icp" and config.dtype == "float32"
                              else "float64"),
            "implementation": "tensor" if backend == "open3d" and method == "icp" else
                              "legacy" if backend == "open3d" else "torchpcl",
            "transform_dtype": "float64" if backend == "open3d" else config.dtype,
        },
    )
