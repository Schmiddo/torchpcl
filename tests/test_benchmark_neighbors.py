"""Neighbor benchmark preparation: subsampling, the radius guard, lazy answers."""

import numpy as np
import pytest
import torch

import torchpcl as tp
from benchmarks.neighbors_cases import (
    MAX_NEIGHBORS,
    NeighborsConfig,
    check_radius_capacity,
    make_case,
    prepare_neighbors,
)


@pytest.fixture(scope="module")
def prepared():
    return prepare_neighbors(
        NeighborsConfig(num_reference=2_000, num_queries=256, k=8, radius=0.15))


def test_radius_search_counts_match_brute_force(prepared):
    reference = torch.tensor(prepared.reference)
    queries = torch.tensor(prepared.queries[:prepared.gate])

    found = tp.radius_neighbors(reference, queries, prepared.config.radius,
                                max_neighbors=MAX_NEIGHBORS).valid.sum(1)

    np.testing.assert_array_equal(found.numpy(), prepared.radius_counts())


def test_radius_guard_rejects_a_truncating_radius():
    # A radius this wide overruns torchpcl's fixed-width result, which would
    # compare a truncated neighborhood against a complete one.
    prepared = prepare_neighbors(
        NeighborsConfig(num_reference=5_000, num_queries=64, k=8, radius=0.5))

    assert prepared.max_radius_count > MAX_NEIGHBORS
    with pytest.raises(ValueError, match="above torchpcl's cap"):
        check_radius_capacity(prepared)
    with pytest.raises(ValueError, match="above torchpcl's cap"):
        make_case(prepared, "radius", "torchpcl")


def test_brute_force_answers_are_computed_only_when_asked(prepared):
    fresh = prepare_neighbors(prepared.config)
    assert (fresh._distances2, fresh._counts) == (None, None)

    make_case(fresh, "knn", "torchpcl")

    # A timing run measures a search; it does not check one.
    assert (fresh._distances2, fresh._counts) == (None, None)

    make_case(fresh, "radius", "torchpcl")

    # The capacity guard needs counts, but never the k nearest distances.
    assert fresh._counts is not None
    assert fresh._distances2 is None


def test_supplied_cloud_becomes_reference_with_subsampled_queries():
    points = np.random.default_rng(0).random((500, 3))

    prepared = prepare_neighbors(
        NeighborsConfig(num_queries=64, k=4, radius=0.2), points)

    assert prepared.source == "fragment"
    assert prepared.reference.shape == (500, 3)
    # Queries are drawn from the cloud itself, as normals and FPFH do.
    assert prepared.queries.shape == (64, 3)
    assert np.isin(prepared.queries, prepared.reference).all()
