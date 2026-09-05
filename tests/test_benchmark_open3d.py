"""Exercise the actual tensor ICP path and verify device/precision at its boundary."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.cases import PreparedPair, RegistrationConfig, make_case
from benchmarks.runner import measure


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_open3d_icp_uses_tensor_inputs(monkeypatch, dtype, device):
    o3d = pytest.importorskip("open3d")
    if device == "cuda" and not o3d.core.cuda.is_available():
        pytest.skip("Open3D CUDA unavailable")
    points = np.random.default_rng(7).normal(size=(32, 3)).astype(dtype)
    expected = np.eye(4)
    expected[:3, 3] = [0.01, -0.02, 0.03]
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points + expected[:3, 3]))
    pair = PreparedPair(Path("synthetic"), RegistrationConfig(dtype=dtype), expected,
                        (source, target), (source, target), None)
    original_icp = o3d.t.pipelines.registration.icp
    calls = []

    def checked_icp(source, target, *args, **kwargs):
        expected_dtype = o3d.core.Dtype.Float32 if dtype == "float32" else o3d.core.Dtype.Float64
        for cloud in (source, target):
            assert cloud.point.positions.dtype == expected_dtype
            assert str(cloud.point.positions.device) == device.upper() + ":0"
        assert kwargs["voxel_size"] == -1.0
        calls.append(True)
        return original_icp(source, target, *args, **kwargs)

    monkeypatch.setattr(o3d.t.pipelines.registration, "icp", checked_icp)
    case = make_case(pair, "icp", "open3d", device)
    report = measure(case, repeats=1, warmup=0)
    # One timed execution and one separate peak-memory execution.
    assert calls == [True, True]
    assert report["implementation"] == "tensor"
    assert not report["reference_only"]
    assert report["median"]["source_displacement_rmse"] < 1e-5
    assert report["input_dtype"] == dtype
    assert report["device"] == device
