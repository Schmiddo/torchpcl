"""Redwood record matching, transform direction, and reference error scoring."""

import numpy as np
import pytest

from benchmarks.redwood_cases import load_ground_truth, transformation_error, summarize


def write_blocks(path, records):
    with path.open("w") as stream:
        for header, matrix in records:
            stream.write(" ".join(map(str, header)) + "\n")
            np.savetxt(stream, matrix)


def test_gt_matches_headers_and_excludes_consecutive(tmp_path):
    expected = np.eye(4)
    expected[0, 3] = 2
    write_blocks(tmp_path / "gt.log", [((0, 1, 3), np.eye(4)), ((0, 2, 3), expected)])
    write_blocks(tmp_path / "gt.info", [((0, 2, 3), 7*np.eye(6)), ((0, 1, 3), np.eye(6))])
    pair, = load_ground_truth(tmp_path)
    assert (pair.source, pair.target, pair.fragments) == (2, 0, 3)
    np.testing.assert_array_equal(pair.transform, expected)
    np.testing.assert_array_equal(pair.information, 7*np.eye(6))


def test_mismatched_gt_headers_rejected(tmp_path):
    write_blocks(tmp_path / "gt.log", [((0, 2, 4), np.eye(4))])
    write_blocks(tmp_path / "gt.info", [((0, 3, 4), np.eye(6))])
    with pytest.raises(ValueError, match="headers differ"):
        load_ground_truth(tmp_path)


def test_error_uses_relative_transform_and_translation_first():
    expected = np.eye(4)
    expected[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    expected[:3, 3] = [3, 4, 5]
    delta = np.eye(4)
    delta[0, 3] = 0.2
    info = np.diag([10., 20, 30, 40, 50, 60])
    assert transformation_error(expected, expected, info) == pytest.approx(0)
    assert transformation_error(expected @ delta, expected, info) == pytest.approx(0.04)
    assert transformation_error(np.linalg.inv(expected), expected, info) > 0.04


@pytest.mark.parametrize("angle", [0.1, 1.5, 2.8, np.pi])
def test_quaternion_scoring_with_translation_rotation_cross_terms(angle):
    estimate = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    estimate[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    estimate[0, 3] = 0.1
    info = np.eye(6)
    info[0, 5] = info[5, 0] = 0.3
    # Reference MATLAB uses minus the vector part of its passive quaternion,
    # hence the positive active-rotation sin(angle/2) here. No factor of two.
    qz = np.sin(angle / 2)
    expected = 0.1**2 + qz**2 + 2 * 0.3 * 0.1 * qz
    assert transformation_error(estimate, np.eye(4), info) == pytest.approx(expected)


def test_summary_retains_per_repeat_recall():
    rows = [{"backend": "torchpcl", "device": "cpu", "median": {"time_ms": 2},
             "samples": [{"registration_success": a}, {"registration_success": b}]}
            for a, b in [(True, False), (True, True)]]
    summary, = summarize(rows)
    assert summary["successful_pairs_per_repeat"] == [2, 1]
    assert summary["registration_recall_per_repeat"] == [1., 0.5]
    assert summary["mean_registration_recall"] == 0.75
