"""Accuracy-tool scoring: pure array comparisons, checkable without Open3D."""

import numpy as np
import pytest

from benchmarks.accuracy import (
    CHECKABLE_OPS,
    Check,
    centroid_keys,
    centroid_mismatch,
    descriptor_mismatch,
    distance_error,
    mask_mismatch,
    normal_deviation,
)


def test_comparisons_are_insensitive_to_what_is_not_contractual():
    # Neighbor order is unspecified, so only the multiset of distances counts.
    assert distance_error(np.array([[4.0, 1.0]]), np.array([[1.0, 4.0]])) == 0
    assert distance_error(np.array([[1.0, 4.5]]), np.array([[1.0, 4.0]])) == 0.5
    # Normal sign is arbitrary without attached normals; direction is not.
    flipped = normal_deviation(np.array([[1.0, 0, 0]]), np.array([[-1.0, 0, 0]]))
    assert flipped == pytest.approx(0)
    assert normal_deviation(np.array([[1.0, 0, 0]]),
                            np.array([[0.0, 1, 0]])) == pytest.approx(1)


def test_centroid_and_mask_comparisons_report_fractions():
    expected = centroid_keys(np.array([[0.0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]))
    found = centroid_keys(np.array([[0.0, 0, 0], [1, 1, 1], [2, 2, 2]]))
    assert centroid_mismatch(found, expected) == 0.25
    # Below the quantization a centroid is the same voxel, above it is not.
    assert centroid_mismatch(centroid_keys(np.array([[1e-7, 0, 0]])),
                             centroid_keys(np.array([[0.0, 0, 0]]))) == 0
    assert mask_mismatch(np.array([True, True, False, True]),
                         np.array([True, True, True, True])) == 0.25


def test_descriptor_mismatch_counts_rows_and_skips_invalid_ones():
    expected = np.zeros((4, 33))
    found = np.zeros((4, 33))
    found[1, 0] = 0.5    # within one bin of mass
    found[2, 0] = 2.0    # a genuinely different neighborhood
    found[3, 0] = 100.0  # invalid row, must not be scored
    valid = np.array([True, True, True, False])

    fraction, worst = descriptor_mismatch(found, expected, valid)

    assert fraction == pytest.approx(1 / 3)
    assert worst == 2.0
    assert descriptor_mismatch(found, expected, np.zeros(4, bool)) == (0.0, 0.0)


def test_check_bounds_separate_gates_from_context():
    def make(bound, value, limit):
        return Check("fpfh", "cpu", "m", value, bound, limit, "open3d-cpu", "why")

    assert make("max", 0.5, 1.0).passed and not make("max", 1.5, 1.0).passed
    assert make("min", 1.5, 1.0).passed and not make("min", 0.5, 1.0).passed
    # Context rows report a number without judging it, and carry no limit.
    info = make("info", 1e9, None)
    assert info.passed and info.row()["limit"] is None


def test_only_comparable_operations_are_checked():
    # An index, a repeated kNN and a bare voxelization produce nothing the
    # other checks do not already compare.
    assert set(CHECKABLE_OPS).isdisjoint(
        {"index-build", "knn-build-query", "voxelize"})
