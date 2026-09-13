"""Preprocessing cases: the measured parameter mappings and the accuracy gates."""

import numpy as np
import pytest

from benchmarks.preprocess_cases import (
    PreprocessConfig,
    make_case,
    prepare_cloud,
    reference_output,
)
from benchmarks.runner import measure

pytest.importorskip("psutil")
pytest.importorskip("open3d")


@pytest.fixture(scope="module")
def prepared():
    points = (np.random.default_rng(0).random((5_000, 3))
              .astype("float32").astype(np.float64))
    return prepare_cloud(
        PreprocessConfig(voxel_size=0.05, normal_k=30, outlier_k=20,
                         outlier_radius=0.1), points)


def _score(prepared, op, metric):
    reference = reference_output(prepared, op)
    report = measure(
        make_case(prepared, op, "torchpcl", reference=reference),
        repeats=1, warmup=0)
    return report["median"][metric]


def test_voxelize_reduce_reproduces_open3d(prepared):
    # voxelize alone returns no centroids; only the reduction is comparable.
    assert _score(prepared, "voxelize-reduce", "centroid_mismatch_fraction") == 0


@pytest.mark.parametrize("op", ["statistical-outliers", "radius-outliers"])
def test_outlier_neighbor_count_maps_directly(prepared, op):
    # torchpcl's k is Open3D's nb_neighbors / nb_points with no off-by-one,
    # despite torchpcl counting the query point: verified discriminatingly,
    # k±1 disagrees. This pins the mapping the benchmark relies on.
    assert _score(prepared, op, "mask_mismatch_fraction") == 0


def test_normals_match_open3d_up_to_sign(prepared):
    # torchpcl's signs are arbitrary without attached normals, so the gate
    # scores unsigned alignment.
    assert _score(prepared, "normals", "min_normal_alignment") > 0.99
