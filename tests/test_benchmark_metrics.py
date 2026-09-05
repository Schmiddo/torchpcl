"""Check benchmark scores independently of either registration implementation."""

import numpy as np
import pytest


from benchmarks import metrics as benchmark


def test_pose_errors_known_rotation_and_translation():
    estimate = np.array([[0., -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 0], [0, 0, 0, 1]])
    scores = benchmark.pose_errors(estimate, np.eye(4), np.array([[1., 0, 0]]))
    assert scores["rotation_error_deg"] == pytest.approx(90)
    assert scores["translation_error"] == pytest.approx(5)
    assert scores["source_displacement_rmse"] == pytest.approx(np.sqrt(29))


def test_pose_errors_exact_rounded_ground_truth():
    angle = 0.1
    expected = np.eye(4)
    expected[:2, :2] = np.round([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]], 6)
    expected[:3, 3] = [1, 2, 3]
    scores = benchmark.pose_errors(expected, expected, np.ones((5, 3)))
    assert scores == pytest.approx({"rotation_error_deg": 0,
                                    "translation_error": 0,
                                    "source_displacement_rmse": 0}, abs=1e-5)
    inverse_scores = benchmark.pose_errors(np.linalg.inv(expected), expected, np.ones((5, 3)))
    assert inverse_scores["translation_error"] > 1


@pytest.mark.parametrize("matrix", [np.eye(3), np.diag([2, 1, 1, 1]),
                                    np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)])
def test_invalid_ground_truth_is_rejected(tmp_path, matrix):
    path = tmp_path / "pose.txt"
    np.savetxt(path, matrix)
    with pytest.raises(ValueError):
        benchmark.load_transform(path)
