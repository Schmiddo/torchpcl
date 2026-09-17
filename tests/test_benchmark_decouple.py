"""Performance entry points and offline comparison, without accuracy scoring."""

import builtins
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import compare, neighbors_cases, operations, sampling
from benchmarks.case import Case
from benchmarks.features_cases import FeaturesConfig, prepare_features
from benchmarks.inputs import load_fragment, random_normals
from benchmarks.runner import measure


COMMON = ["--device", "cpu", "--threads", "1", "--repeats", "1", "--warmup", "0"]
SMALL_OPS = ["--num-reference", "128", "--num-queries", "16", "--k", "8",
             "--normal-k", "8", "--outlier-k", "8", "--feature-radius", "0.3"]


def forbid_baselines(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("open3d", "small_gicp"):
            raise AssertionError(f"unexpected baseline import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_performance_skips_scoring_but_default_keeps_it(monkeypatch):
    scored = []
    monkeypatch.setattr("benchmarks.runner.sample_peak_memory", lambda case: {})
    case = Case("test", "torchpcl", "cpu", run=lambda: 42,
                evaluate=lambda value: scored.append(value) or {"score": value})
    report = measure(case, repeats=2, warmup=1, evaluate=False)
    assert scored == []
    assert all(set(sample) == {"time_ms"} for sample in report["samples"])
    assert measure(case, repeats=1, warmup=0)["median"]["score"] == 42
    assert scored == [42]


def test_torchpcl_entry_points_without_baselines(monkeypatch, tmp_path, search_device):
    pytest.importorskip("psutil")
    forbid_baselines(monkeypatch)
    common = COMMON.copy()
    common[1] = search_device.type
    path = tmp_path / "operations.json"
    report = operations.main(common + SMALL_OPS + ["--json", str(path)])
    assert json.loads(path.read_text()) == report
    assert {row["name"] for row in report["results"]} == set(operations.ALL_OPS)
    assert report["config"]["normals_source"] == "synthetic"
    sampled = sampling.main(common + ["--num-points", "128", "--num-samples", "8"])
    for result in (report, sampled):
        assert result["environment"]["open3d"] is None
        assert result["config"]["backend"] == "torchpcl"
        assert all(row["backend"] == "torchpcl" for row in result["results"])
        assert all(set(row["median"]) == {"time_ms"} for row in result["results"])


@pytest.mark.parametrize("op,brute_forced", [("knn", False), ("radius", True)])
def test_timing_runs_brute_force_only_for_the_radius_guard(monkeypatch, op, brute_forced):
    """A timing run measures a search; only the radius cap guard needs answers.

    Brute forcing the scored prefix costs seconds at benchmark sizes, and a
    performance run has nothing to do with the result.
    """
    pytest.importorskip("psutil")
    calls = []
    original = neighbors_cases.PreparedNeighbors._blocks
    monkeypatch.setattr(neighbors_cases.PreparedNeighbors, "_blocks",
                        lambda self: calls.append(op) or original(self))

    operations.main(COMMON + SMALL_OPS + ["--op", op])

    assert bool(calls) is brute_forced


def write_ply(path, normals=False):
    from plyfile import PlyData, PlyElement

    rng = np.random.default_rng(2)
    properties = ["x", "y", "z"] + (["nx", "ny", "nz"] if normals else [])
    vertices = np.zeros(32, dtype=[(name, "f4") for name in properties])
    for name in ("x", "y", "z"):
        vertices[name] = rng.random(32)
    if normals:
        vertices["nz"] = 1
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
    return np.column_stack([vertices[name] for name in ("x", "y", "z")])


def test_ply_and_explicit_normals_without_open3d(monkeypatch, tmp_path):
    pytest.importorskip("psutil")
    forbid_baselines(monkeypatch)
    path = tmp_path / "cloud.ply"
    points = write_ply(path)
    loaded_points, loaded_normals = load_fragment(path, "float32")
    np.testing.assert_array_equal(loaded_points, points)
    assert loaded_normals is None
    args = COMMON + ["--input", "redwood", "--fragment", str(path)]
    assert operations.main(args + ["--op", "voxelize"])["results"]
    with pytest.raises(SystemExit, match="--normals"):
        operations.main(args + ["--op", "fpfh"])
    normals_path = tmp_path / "normals.npy"
    np.save(normals_path, random_normals(len(points), 0))
    report = operations.main(args + ["--op", "fpfh", "--normals", str(normals_path)])
    assert report["config"]["normals_source"] == "npy"
    assert report["config"]["normals"] == str(normals_path)
    write_ply(path, normals=True)
    import torchpcl as tp

    load_ply = tp.load_ply
    loaded_clouds = []

    def record_load(*args, **kwargs):
        cloud = load_ply(*args, **kwargs)
        loaded_clouds.append(cloud)
        return cloud

    monkeypatch.setattr(tp, "load_ply", record_load)
    assert operations.main(args + ["--op", "fpfh"])["config"]["normals_source"] == "ply"
    assert len(loaded_clouds) == 1
    np.testing.assert_array_equal(loaded_clouds[0].points.numpy(), points)
    np.testing.assert_array_equal(loaded_clouds[0].normals.numpy(),
                                  np.tile([0, 0, 1], (len(points), 1)))


@pytest.mark.parametrize("normals", [np.ones((2, 3)), np.zeros((3, 3)),
                                      np.full((3, 3), np.nan), np.full((3, 3), np.inf)])
def test_explicit_normals_validation(normals):
    with pytest.raises(ValueError, match="FPFH normals"):
        prepare_features(FeaturesConfig(), np.ones((3, 3)), normals=normals)


def test_normals_are_repeatable_and_quantized_without_open3d(monkeypatch):
    forbid_baselines(monkeypatch)
    normals = random_normals(32, 0)
    np.testing.assert_array_equal(normals, random_normals(32, 0))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    assert not np.array_equal(normals, random_normals(32, 1))
    prepared = prepare_features(FeaturesConfig(), np.ones((32, 3)), normals=normals)
    np.testing.assert_array_equal(prepared.normals, normals.astype("float32").astype("float64"))


def test_missing_selected_backend_and_legacy_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(SystemExit, match="Open3D is required"):
        operations.main(COMMON + ["--backend", "open3d"])
    with pytest.raises(SystemExit):
        sampling.parse_args(["--backend", "open3d-legacy", "--device", "cuda"])


def test_cross_backend_comparison_offline(tmp_path):
    def row(name, backend, time):
        return {"name": name, "backend": backend, "device": "cpu",
                "median": {"time_ms": time}, "time_ms_min": time, "time_ms_max": time}

    before = {"environment": {"open3d": "0.19", "machine": "baseline-host"},
              "config": {"seed": 0},
              "results": [row("knn", "open3d", 10), row("knn", "torchpcl", 99),
                          row("radius", "open3d", 2)]}
    after = {"environment": {"torchpcl": "0.2", "machine": "candidate-host"},
             "config": {"seed": 1},
             "results": [row("knn", "torchpcl", 5), row("normals", "torchpcl", 1)]}
    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path, report in zip(paths, (before, after)):
        path.write_text(json.dumps(report))
    # -S excludes site-packages: comparison needs neither backend nor NumPy.
    command = [sys.executable, "-S", str(Path(compare.__file__)), *map(str, paths),
               "--before-backend", "open3d", "--after-backend", "torchpcl"]
    output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    assert "2.00x" in output
    assert "baseline-host" in output and "candidate-host" in output
    assert "open3d=0.19" in output and "! config seed: 0 -> 1" in output
    assert "radius/cpu" in output and "normals/cpu" in output and "only in" in output
    with pytest.raises(SystemExit, match="no results for backend"):
        compare.main([*map(str, paths), "--before-backend", "missing", "--after-backend", "torchpcl"])


@pytest.mark.parametrize("options", [["--before-backend", "open3d"],
                                      ["--backend", "torchpcl", "--before-backend", "open3d",
                                       "--after-backend", "torchpcl"]])
def test_comparison_requires_unambiguous_selectors(options):
    with pytest.raises(SystemExit):
        compare.parse_args(["before.json", "after.json", *options])
