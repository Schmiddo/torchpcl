import json

import pytest

from benchmarks.cli import source_revision
from benchmarks.compare import main


def write_report(path, rows, **config):
    report = {
        "environment": {"timestamp": "2026-09-13T16:00:00+00:00", "revision": "abc123"},
        "config": {"dtype": "float32", **config},
        "results": [
            {"name": name, "backend": "torchpcl", "device": "cuda",
             "time_ms_min": low, "time_ms_max": high,
             "median": {"time_ms": median, **metrics}}
            for name, low, median, high, metrics in rows
        ],
    }
    path.write_text(json.dumps(report))
    return path


def test_reports_change_and_marks_only_separated_rows(tmp_path, capsys):
    before = write_report(tmp_path / "a.json", [
        ("knn", 9.0, 10.0, 11.0, {}),
        ("radius", 5.0, 10.0, 15.0, {}),
    ])
    after = write_report(tmp_path / "b.json", [
        ("knn", 4.0, 5.0, 6.0, {}),
        ("radius", 4.0, 8.0, 16.0, {}),
    ])
    main([str(before), str(after)])
    lines = capsys.readouterr().out.splitlines()
    assert "abc123" in lines[0]
    # knn halves with disjoint ranges; radius moves inside overlapping spread.
    assert [line for line in lines if line.startswith("knn")] == [
        f"{'knn/torchpcl/cuda':40} {10.0:10.3f} {5.0:10.3f}    -50.0%*"]
    assert [line for line in lines if line.startswith("radius")] == [
        f"{'radius/torchpcl/cuda':40} {10.0:10.3f} {8.0:10.3f}    -20.0% "]


def test_flags_config_drift_gate_movement_and_unmatched_rows(tmp_path, capsys):
    before = write_report(tmp_path / "a.json", [
        ("knn", 1.0, 1.0, 1.0, {"max_distance2_error": 0.0}),
        ("fpfh", 1.0, 1.0, 1.0, {}),
    ], radius=0.02)
    after = write_report(tmp_path / "b.json", [
        ("knn", 1.0, 1.0, 1.0, {"max_distance2_error": 0.5}),
        ("normals", 1.0, 1.0, 1.0, {}),
    ], radius=0.01)
    main([str(before), str(after)])
    out = capsys.readouterr().out
    assert "! config radius: 0.02 -> 0.01" in out
    assert "gate: max_distance2_error 0 -> 0.5" in out
    assert f"{'fpfh/torchpcl/cuda':40} only in {before}" in out
    assert f"{'normals/torchpcl/cuda':40} only in {after}" in out


def test_grouped_rows_are_keyed_by_their_input(tmp_path, capsys):
    """Registration and Redwood measure the same case on several inputs; the
    group keeps those rows apart."""
    report = {
        "config": {}, "environment": {},
        "results": [
            {"name": "icp", "group": group, "backend": "torchpcl", "device": "cpu",
             "time_ms_min": 1.0, "time_ms_max": 1.0, "median": {"time_ms": 1.0}}
            for group in ("office1", "office2")],
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(report))
    main([str(path), str(path)])
    out = capsys.readouterr().out
    assert "office1/icp/torchpcl/cpu" in out and "office2/icp/torchpcl/cpu" in out


def test_rejects_files_that_are_not_reports(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"results": []}))
    with pytest.raises(SystemExit, match="not a benchmark report"):
        main([str(path), str(path)])


def test_source_revision_identifies_the_working_tree():
    revision = source_revision()
    assert revision is None or revision.split("-")[0].isalnum()
