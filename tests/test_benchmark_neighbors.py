"""Neighbor-search cases: brute-force scoring, device placement, radius cap."""

import pytest

from benchmarks.neighbors_cases import (
    MAX_NEIGHBORS,
    NeighborsConfig,
    make_case,
    prepare_neighbors,
)
from benchmarks.runner import measure

# measure() samples process memory; that lives in the benchmark group.
pytest.importorskip("psutil")


@pytest.fixture(scope="module")
def prepared():
    return prepare_neighbors(
        NeighborsConfig(num_reference=2_000, num_queries=256, k=8, radius=0.15))


def test_knn_case_matches_brute_force(prepared):
    report = measure(make_case(prepared, "knn", "torchpcl"), repeats=1, warmup=0)

    # float32 coordinates, so agreement is exact only to rounding.
    assert report["median"]["max_distance2_error"] < 1e-6


def test_radius_case_counts_match_brute_force(prepared):
    report = measure(make_case(prepared, "radius", "torchpcl"), repeats=1, warmup=0)

    assert report["median"]["count_mismatch_fraction"] == 0


def test_radius_case_rejects_neighborhoods_over_the_cap():
    # A radius this wide truncates torchpcl's fixed-width result, which would
    # score a truncated neighborhood against a complete one.
    prepared = prepare_neighbors(
        NeighborsConfig(num_reference=5_000, num_queries=64, k=8, radius=0.5))

    assert prepared.max_radius_count > MAX_NEIGHBORS
    with pytest.raises(ValueError, match="above torchpcl's cap"):
        make_case(prepared, "radius", "torchpcl")


def test_supplied_cloud_becomes_reference_with_subsampled_queries():
    import numpy as np

    points = np.random.default_rng(0).random((500, 3))

    prepared = prepare_neighbors(
        NeighborsConfig(num_queries=64, k=4, radius=0.2), points)

    assert prepared.source == "fragment"
    assert prepared.reference.shape == (500, 3)
    # Queries are drawn from the cloud itself, as normals and FPFH do.
    assert prepared.queries.shape == (64, 3)
    assert np.isin(prepared.queries, prepared.reference).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_open3d_knn_case_runs_on_requested_device(prepared, device):
    o3d = pytest.importorskip("open3d")
    if device == "cuda" and not o3d.core.cuda.is_available():
        pytest.skip("Open3D CUDA unavailable")

    report = measure(
        make_case(prepared, "knn", "open3d", device), repeats=1, warmup=0)

    assert report["device"] == device
    assert report["median"]["max_distance2_error"] < 1e-6
