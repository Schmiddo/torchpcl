from types import SimpleNamespace

import pytest

from benchmarks.case import select_devices
from benchmarks.registration import parse_args
from benchmarks.registration_cases import RegistrationConfig
from benchmarks.runner import measure


def test_default_devices_and_precision():
    args = parse_args([])
    assert args.device == "all"
    assert args.dtype == RegistrationConfig().dtype == "float32"
    assert args.voxel_size == RegistrationConfig().voxel_size == 0.05
    assert args.threads >= 1
    assert select_devices("all", True) == (["cpu", "cuda"], [])
    devices, skipped = select_devices("all", False)
    assert devices == ["cpu"]
    assert skipped[0]["device"] == "cuda"
    assert select_devices("cpu", False) == (["cpu"], [])
    with pytest.raises(ValueError, match="CUDA"):
        select_devices("cuda", False)


def test_runner_synchronizes_and_excludes_setup_and_evaluation(monkeypatch):
    events = []
    ticks = iter([1., 2., 3., 5., 6., 9.])

    def clock():
        events.append("clock")
        return next(ticks)

    def evaluate(result):
        assert result == "result"
        events.append("evaluate")
        return dict(rotation_error_deg=1, translation_error=2,
                    source_displacement_rmse=3, fitness=0.9, inlier_rmse=0.1)

    case = SimpleNamespace(
        name="icp", backend="torchpcl", device="cuda",
        metadata={"input_dtype": "float32"},
        device_memory_reader=None,
        reset=lambda: events.append("reset"),
        synchronize=lambda: events.append("sync"),
        run=lambda: (events.append("run"), "result")[1],
        evaluate=evaluate,
    )
    monkeypatch.setattr("benchmarks.runner.perf_counter", clock)
    monkeypatch.setattr(
        "benchmarks.runner.sample_peak_memory",
        lambda case: {"process_rss_peak_mib": 100, "device_peak_mib": 20},
    )
    result = measure(case, repeats=2, warmup=1)
    timed = ["reset", "sync", "clock", "run", "sync", "clock"]
    assert events == timed + (timed + ["evaluate"]) * 2
    assert [sample["time_ms"] for sample in result["samples"]] == [2000, 3000]
    assert result["median"]["time_ms"] == 2500
    assert (result["time_ms_min"], result["time_ms_max"]) == (2000, 3000)
    assert result["device"] == "cuda"
    assert result["input_dtype"] == "float32"
    assert result["memory"]["device_peak_mib"] == 20
