"""Registration preparation and scoring, both independent of Open3D."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.registration_cases import (
    FEATURE_NEIGHBORS,
    PreparedPair,
    RegistrationConfig,
    make_case,
    prepare_cloud,
)

DATA = Path("data")


@pytest.fixture(scope="module")
def prepared():
    path = DATA / "source.ply"
    if not path.is_file():
        pytest.skip(f"{path} missing; run `just download-data`")
    config = RegistrationConfig(voxel_size=0.05)
    original, points, features = prepare_cloud(path, config)
    return config, original, points, features


def test_preparation_yields_quantized_arrays(prepared):
    config, original, points, features = prepared

    # Plain arrays, so each adapter builds its own objects from them.
    for array in (original, points, features):
        assert isinstance(array, np.ndarray) and array.dtype == np.float64
    assert points.shape[1] == 3 and len(points) < len(original)
    assert features.shape == (len(points), 33)
    # Quantized to the configured dtype, then held as float64, so both
    # backends start from identical coordinate values.
    np.testing.assert_array_equal(points, points.astype(config.dtype))


def test_reported_preprocessing_identifies_the_workload(prepared):
    config, original, points, features = prepared
    pair = PreparedPair(DATA, config, np.eye(4), (original, original),
                        (points, points), (features, features))

    preprocessing = pair.metadata()["preprocessing"]

    # Preparation is untimed and not a measurement, so what matters is that
    # the choice is recorded rather than that it matches another library.
    assert preprocessing["provider"] == "torchpcl"
    assert preprocessing["feature_neighbors"] == FEATURE_NEIGHBORS
    assert preprocessing["feature_radius"] == 5 * config.voxel_size


def test_scoring_needs_no_baseline_unless_asked(prepared):
    config, original, points, features = prepared
    pair = PreparedPair(DATA, config, np.eye(4), (original, original),
                        (points, points), (features, features))

    case = make_case(pair, "icp", "torchpcl", "cpu")
    metrics = case.evaluate(case.run_synchronized())

    # Pose error is scored against ground truth; fitness and inlier RMSE need
    # Open3D and are therefore opt-in.
    assert "rotation_error_deg" in metrics
    assert "fitness" not in metrics and "inlier_rmse" not in metrics
