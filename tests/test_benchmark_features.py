"""FPFH cases: agreement with Open3D and the neighbor-truncation diagnostic."""

import numpy as np
import pytest

from benchmarks.features_cases import (
    FeaturesConfig,
    make_case,
    prepare_features,
    reference_output,
)
from benchmarks.runner import measure

pytest.importorskip("psutil")
o3d = pytest.importorskip("open3d")


def _prepare(radius):
    points = (np.random.default_rng(0).random((5_000, 3))
              .astype("float32").astype(np.float64))
    cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(points.astype("float32")))
    cloud.estimate_normals(max_nn=30)
    cloud.orient_normals_towards_camera_location(o3d.core.Tensor(
        np.zeros(3, dtype="float32")))
    return prepare_features(FeaturesConfig(radius=radius), points,
                            cloud.point.normals.numpy())


def test_fpfh_reproduces_open3d():
    prepared = _prepare(0.1)

    report = measure(
        make_case(prepared, "fpfh", "torchpcl",
                  reference=reference_output(prepared)),
        repeats=1, warmup=0)

    assert report["median"]["descriptor_mismatch_fraction"] == 0
    assert report["median"]["valid_fraction"] == 1


def test_fpfh_reports_neighbor_truncation():
    # A radius holding far more than the 64-neighbor cap: both libraries
    # summarize a truncated neighborhood, and the run says so rather than
    # letting the comparison look complete.
    prepared = _prepare(0.2)

    report = measure(
        make_case(prepared, "fpfh", "torchpcl",
                  reference=reference_output(prepared)),
        repeats=1, warmup=0)

    assert report["median"]["limit_reached_fraction"] > 0
