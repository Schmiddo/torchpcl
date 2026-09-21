"""The history store and the dashboard it feeds.

These cover the decisions that make a saved point trustworthy -- what may share
a series, what is refused outright -- rather than re-testing the drivers.
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from benchmarks import cli, dashboard, history


def write_report(path, *, revision="abc1234", machine="box", results=None, config=None):
    environment = {"revision": revision, "timestamp": "2026-09-17T10:00:00+00:00",
                   "python": "3.14.3", "torch": "2.12.0", "torchpcl": "0.2.0"}
    if machine is not None:
        environment["machine"] = machine
    report = {
        "environment": environment,
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


def test_a_machine_label_describes_the_hardware_rather_than_the_host():
    """A kept history may be shared, so the label must not be a hostname."""
    assert cli.machine_label() != cli.platform.node().lower()
    assert cli.machine_label() == cli.slug(cli.machine_label())


def test_a_report_without_a_machine_is_kept_rather_than_refused(tmp_path):
    """An older report still carries usable timings; the missing label only
    means its points cannot be told apart from another unlabelled computer's."""
    report = write_report(tmp_path / "run.json", machine=None)
    store = tmp_path / "history"
    history.main(["--history", str(store), "add", str(report)])
    stored = history.read_history(store / f"{history.UNKNOWN_MACHINE}.jsonl")
    assert [row["machine"] for row in stored] == [history.UNKNOWN_MACHINE]


def test_relabelling_moves_a_machine_but_refuses_to_merge_two(tmp_path):
    store = tmp_path / "history"
    history.main(["--history", str(store), "add",
                  str(write_report(tmp_path / "a.json", machine="oldname"))])
    history.main(["--history", str(store), "add",
                  str(write_report(tmp_path / "b.json", machine="other",
                                   revision="bbb2222"))])

    history.main(["--history", str(store), "relabel", "oldname", "linux-some-cpu"])
    assert not (store / "oldname.jsonl").exists()
    rows = history.read_history(store / "linux-some-cpu.jsonl")
    assert [row["machine"] for row in rows] == ["linux-some-cpu"]

    with pytest.raises(SystemExit, match="merging two machines"):
        history.main(["--history", str(store), "relabel", "other", "linux-some-cpu"])


def test_the_served_page_shows_points_kept_after_the_server_started(tmp_path):
    """The whole point of serving rather than writing a file: keep a point,
    reload, see it."""
    store = tmp_path / "history"
    history.main(["--history", str(store), "add",
                  str(write_report(tmp_path / "a.json", revision="aaa1111"))])
    args = dashboard.parse_args(["--history", str(store), "--port", "0",
                                 "--host", "127.0.0.1"])
    server = ThreadingHTTPServer((args.host, 0), dashboard.handler_class(args))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        first = urlopen(url).read().decode()
        assert "aaa1111" in first and "bbb2222" not in first

        history.main(["--history", str(store), "add",
                      str(write_report(tmp_path / "b.json", revision="bbb2222"))])
        assert "bbb2222" in urlopen(url).read().decode()

        with pytest.raises(HTTPError) as missing:
            urlopen(url + "elsewhere")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
