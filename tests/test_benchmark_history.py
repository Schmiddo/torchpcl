"""The history store and the dashboard it feeds.

These cover the decisions that make a saved point trustworthy -- what may share
a series, what is refused outright -- rather than re-testing the drivers.
"""

from __future__ import annotations

import json

import pytest

from benchmarks import dashboard, history


def write_report(path, *, revision="abc1234", machine="box", results=None, config=None):
    report = {
        "environment": {"revision": revision, "machine": machine,
                        "timestamp": "2026-09-17T10:00:00+00:00",
                        "python": "3.14.3", "torch": "2.12.0", "torchpcl": "0.2.0"},
        "config": {"ops": ["knn"], "input": "random", "threads": 8,
                   "repeats": 5, "devices": ["cpu"], **(config or {})},
        "results": results if results is not None else [
            {"name": "knn", "backend": "torchpcl", "device": "cpu",
             "median": {"time_ms": 10.0}, "time_ms_min": 9.5, "time_ms_max": 10.5,
             "memory": {"process_rss_peak_mib": 700.0, "device_peak_mib": None}},
        ],
    }
    path.write_text(json.dumps(report))
    return path


def test_a_dirty_revision_is_refused_unless_explicitly_allowed(tmp_path):
    report = write_report(tmp_path / "run.json", revision="abc1234-dirty")
    store = tmp_path / "history"

    with pytest.raises(SystemExit, match="uncommitted changes"):
        history.main(["--history", str(store), "add", str(report)])

    history.main(["--history", str(store), "add", str(report), "--allow-dirty"])
    assert len(history.read_history(store / "box.jsonl")) == 1


def test_repeats_share_a_series_but_threads_do_not(tmp_path):
    """Sampling the same workload more times is the same workload; using
    different thread counts is not, and the two must not share a line."""
    base = {"ops": ["knn"], "input": "random", "threads": 8}
    assert (history.config_digest({**base, "repeats": 5})
            == history.config_digest({**base, "repeats": 9}))
    assert (history.config_digest({**base, "repeats": 5})
            != history.config_digest({**base, "threads": 24}))


def test_adding_the_same_run_twice_does_not_duplicate_the_point(tmp_path):
    report = write_report(tmp_path / "run.json")
    store = tmp_path / "history"
    history.main(["--history", str(store), "add", str(report)])
    history.main(["--history", str(store), "add", str(report)])
    assert len(history.read_history(store / "box.jsonl")) == 1


def test_an_accuracy_report_is_rejected_rather_than_plotted(tmp_path):
    """`accuracy.py` shares the operation config but answers a different
    question; a timing history must not absorb it."""
    path = tmp_path / "accuracy.json"
    path.write_text(json.dumps({
        "environment": {"revision": "abc1234", "machine": "box"},
        "config": {"ops": ["knn"], "input": "random"},
        "checks": [], "failed": 0,
    }))
    with pytest.raises(SystemExit, match="accuracy report"):
        history.main(["--history", str(tmp_path / "history"), "add", str(path)])


def test_the_dashboard_keeps_incomparable_configurations_on_separate_cards(tmp_path):
    store = tmp_path / "history"
    history.main(["--history", str(store), "add",
                  str(write_report(tmp_path / "a.json", revision="aaa1111"))])
    history.main(["--history", str(store), "add",
                  str(write_report(tmp_path / "b.json", revision="bbb2222"))])
    history.main(["--history", str(store), "add",
                  str(write_report(tmp_path / "c.json", revision="ccc3333",
                                   config={"threads": 24}))])

    rows = dashboard.load_rows(store, None)
    charts = dashboard.build_charts(rows, history.read_configs(store))

    assert len(charts) == 2, "a thread-count change must start a new card"
    shared = next(chart for chart in charts if len(chart["revisions"]) == 2)
    assert shared["revisions"] == ["aaa1111", "bbb2222"]
    assert [series["backend"] for series in shared["series"]] == ["torchpcl"]

    page = dashboard.render(dashboard.page_data(rows, history.read_configs(store)))
    assert "torchpcl benchmark history" in page
    assert "aaa1111" in page
