from types import SimpleNamespace

import pytest

from benchmarks import bench, run
from benchmarks.case import select_devices
from benchmarks.profiles import PROFILES
from benchmarks.registration_cases import RegistrationConfig
from benchmarks.runner import measure


def test_default_devices_and_precision():
    _, _, args = run.parse_args(["registration"])
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


def test_every_profile_pins_exactly_its_suites_workload():
    """A profile missing an argument would let that argument's default drift
    under a series; `check_profile` runs on every parse, so parsing is the test."""
    for name, profile in PROFILES.items():
        suite, parsed, args = run.parse_args(["--profile", name])
        assert (suite.name, parsed) == (profile.suite, profile)
        assert args.threads == profile.params["threads"]


def test_a_profile_accepts_selection_but_rejects_workload_changes(capsys):
    _, _, args = run.parse_args(["--profile", "operations-random", "--op", "knn",
                                 "--device", "cpu", "--repeats", "9"])
    assert (args.op, args.device, args.repeats, args.k) == ("knn", "cpu", 9, 30)
    for override in (["--k", "16"], ["--threads", "3"], ["--warmup", "0"]):
        with pytest.raises(SystemExit):
            run.parse_args(["--profile", "operations-random", *override])
        assert "pinned" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="belongs to the sampling suite"):
        run.parse_args(["operations", "--profile", "sampling"])


def test_the_sweep_plans_torchpcl_or_baselines_and_refuses_to_keep_a_dirty_tree(
        monkeypatch):
    routine = [profile for profile in PROFILES.values() if profile.routine]
    assert [(p.name, backend) for p, backend in bench.plan(None, False)] == [
        (profile.name, "torchpcl") for profile in routine]
    baselines = bench.plan(["sampling"], True)
    assert [backend for _, backend in baselines] == ["open3d-tensor", "open3d-legacy"]

    monkeypatch.setattr(bench, "source_revision", lambda: "abc1234-dirty")
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(SystemExit, match="committed tree"):
        bench.main(["--keep"])
