"""The history store and the dashboard it feeds.

These cover the decisions that make a kept point trustworthy -- what may share
a series, what is refused outright, what a baseline is -- rather than
re-testing the suites.
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from benchmarks import cli, dashboard, history

PARAMS = {"k": 30, "threads": 8}


def write_report(path, *, revision="abc1234", machine="box", backend="torchpcl",
                 profile="operations-random", version=1, params=PARAMS,
                 median=10.0, open3d=None, timestamp="2026-09-17T10:00:00+00:00"):
    report = {
        "environment": {"revision": revision, "timestamp": timestamp,
                        "machine": machine, "python": "3.14.3", "torch": "2.12.0",
                        "torchpcl": "0.2.0", "open3d": open3d},
        "suite": "operations",
        "profile": profile and {"name": profile, "version": version, "params": params},
        "config": {"backend": backend, **params},
        "results": [
            {"name": "knn", "backend": backend, "device": "cpu",
             "median": {"time_ms": median}, "time_ms_min": median - 0.5,
             "time_ms_max": median + 0.5,
             "memory": {"process_rss_peak_mib": 700.0, "device_peak_mib": None}},
        ],
    }
    path.write_text(json.dumps(report))
    return path


def keep(store, *paths):
    history.main(["--history", str(store), "add", *map(str, paths)])


def test_only_committed_profile_runs_are_kept(tmp_path):
    store = tmp_path / "history"
    with pytest.raises(SystemExit, match="uncommitted changes"):
        keep(store, write_report(tmp_path / "dirty.json", revision="abc1234-dirty"))
    with pytest.raises(SystemExit, match="ad hoc run"):
        keep(store, write_report(tmp_path / "adhoc.json", profile=None))
    accuracy = tmp_path / "accuracy.json"
    accuracy.write_text(json.dumps({"environment": {}, "config": {}, "checks": []}))
    with pytest.raises(SystemExit, match="accuracy report"):
        keep(store, accuracy)
    assert not store.exists()


def test_a_profile_edited_without_a_version_bump_is_refused(tmp_path):
    """A pinned value that moved is a different workload on the same line; a
    newly pinned argument at its old implicit value is the same work."""
    store = tmp_path / "history"
    keep(store, write_report(tmp_path / "a.json", revision="aaa1111"))
    keep(store, write_report(tmp_path / "b.json", revision="bbb2222",
                             params={**PARAMS, "seed": 0}))
    with pytest.raises(SystemExit, match="Bump its version"):
        keep(store, write_report(tmp_path / "c.json", revision="ccc3333",
                                 params={**PARAMS, "k": 16}))
    keep(store, write_report(tmp_path / "d.json", revision="ccc3333",
                             params={**PARAMS, "k": 16}, version=2))
    series = history.read_rows(store / "box" / history.SERIES)
    assert [(row["revision"], row["profile_version"]) for row in series] == [
        ("aaa1111", 1), ("bbb2222", 1), ("ccc3333", 2)]


def test_baselines_are_kept_per_library_version_not_per_revision(tmp_path):
    store = tmp_path / "history"
    for number, (revision, version) in enumerate(
            [("aaa1111", "0.19.0"), ("bbb2222", "0.19.0"), ("bbb2222", "0.20.0")]):
        keep(store, write_report(tmp_path / f"{number}.json", revision=revision,
                                 backend="open3d", open3d=version, median=20.0 + number))
    baselines = history.read_rows(store / "box" / history.BASELINES)
    # Re-measuring 0.19 on a later commit refreshed it rather than adding a point.
    assert [(row["baseline_version"], row["median_ms"]) for row in baselines] == [
        ("0.19.0", 21.0), ("0.20.0", 22.0)]
    assert not (store / "box" / history.SERIES).exists()

    report = write_report(tmp_path / "t.json")
    keep(store, report)
    keep(store, report)
    assert len(history.read_rows(store / "box" / history.SERIES)) == 1


def test_the_dashboard_draws_baselines_as_references_and_hides_superseded_versions(
        tmp_path):
    store = tmp_path / "history"
    keep(store, write_report(tmp_path / "a.json", revision="aaa1111", median=10.0))
    keep(store, write_report(tmp_path / "b.json", revision="bbb2222", median=5.0,
                             timestamp="2026-09-18T10:00:00+00:00"))
    keep(store, write_report(tmp_path / "o.json", revision="bbb2222", backend="open3d",
                             open3d="0.19.0", median=20.0))
    keep(store, write_report(tmp_path / "v2.json", revision="ccc3333", version=2,
                             params={**PARAMS, "k": 16}))

    data = dashboard.page_data(*dashboard.load_rows(store, None))
    current = [chart for chart in data["charts"] if chart["current"]]
    superseded = [chart for chart in data["charts"] if not chart["current"]]
    assert [chart["version"] for chart in current + superseded] == [2, 1]

    chart = superseded[0]
    assert chart["revisions"] == ["aaa1111", "bbb2222"]
    assert [series["backend"] for series in chart["series"]] == ["torchpcl"]
    assert [(ref["backend"], ref["version"], ref["median"])
            for ref in chart["references"]] == [("open3d", "0.19.0", 20.0)]
    assert [(move["ratio"], move["significant"]) for move in data["movement"]] == [
        (-0.5, True)]
    assert "aaa1111" in dashboard.render(data)


def test_a_machine_label_describes_the_hardware():
    assert cli.machine_label() != cli.platform.node().lower()
    assert cli.machine_label() == cli.slug(cli.machine_label())


def test_relabelling_moves_a_machine_but_refuses_to_merge_two(tmp_path):
    store = tmp_path / "history"
    keep(store, write_report(tmp_path / "a.json", machine="oldname"))
    keep(store, write_report(tmp_path / "b.json", machine="other", revision="bbb2222"))

    history.main(["--history", str(store), "relabel", "oldname", "linux-some-cpu"])
    assert not (store / "oldname").exists()
    rows = history.read_rows(store / "linux-some-cpu" / history.SERIES)
    assert [row["machine"] for row in rows] == ["linux-some-cpu"]

    with pytest.raises(SystemExit, match="merging two machines"):
        history.main(["--history", str(store), "relabel", "other", "linux-some-cpu"])


def test_the_served_page_shows_points_kept_after_the_server_started(tmp_path):
    """The whole point of serving rather than writing a file: keep a point,
    reload, see it."""
    store = tmp_path / "history"
    keep(store, write_report(tmp_path / "a.json", revision="aaa1111"))
    args = dashboard.parse_args(["--history", str(store), "--port", "0",
                                 "--host", "127.0.0.1"])
    server = ThreadingHTTPServer((args.host, 0), dashboard.handler_class(args))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        first = urlopen(url).read().decode()
        assert "aaa1111" in first and "bbb2222" not in first

        keep(store, write_report(tmp_path / "b.json", revision="bbb2222"))
        assert "bbb2222" in urlopen(url).read().decode()

        with pytest.raises(HTTPError) as missing:
            urlopen(url + "elsewhere")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
