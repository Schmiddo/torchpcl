"""Render the saved benchmark history as one self-contained HTML page.

`history.py` keeps the points; this turns them into charts. By default it
serves them over HTTP, re-rendering on each request so a newly kept point needs
only a reload -- and so the page can be read from another computer. With
`--out` it writes the same page to a file instead: the data is embedded in it,
so that copy needs no server, no network and no dependencies and can be
attached to a message and still work.

An overview matrix at the top puts each case's newest torchpcl time on every
profile and device side by side, against the fastest baseline; below it the
cards sit in one collapsible section per device, CUDA first, collapsed on load.

One card per measured case of a profile version: torchpcl's series across
revisions, with its measured min/max as a band, because a median alone cannot
distinguish a regression from a busy machine. Baselines are reference values
rather than series -- they do not change with torchpcl's commits -- so each
appears as a dashed line at its newest measurement, labelled with its library
version. Cards of superseded profile versions are hidden unless asked for.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare import moved
from benchmarks.history import BASELINES, DEFAULT_HISTORY, SERIES, machines, read_rows

DEFAULT_OUTPUT = Path("runs/dashboard.html")
#: Unlikely to collide with a development server already on the box -- 8000
#: and 8765 routinely are not.
DEFAULT_PORT = 8770

#: Categorical slots 1-3 of the reference palette, which validate all-pairs in
#: both modes. Assignment is by backend identity, never by position, so a
#: filtered-out series never repaints the survivors. Pairs that would collide
#: never share a card: `open3d-tensor`/`open3d-legacy` appear only in sampling,
#: `small-gicp` only in registration and operations.
SERIES_COLORS = {
    "torchpcl": 1,
    "open3d": 2,
    "open3d-tensor": 2,
    "open3d-legacy": 3,
    "small-gicp": 3,
}
FALLBACK_SLOT = 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--out", type=Path, nargs="?", const=DEFAULT_OUTPUT,
                        help=f"write the page to PATH (default: {DEFAULT_OUTPUT}) "
                             "and exit, instead of serving it")
    parser.add_argument("--host", default="0.0.0.0",
                        help="address to serve on; 127.0.0.1 to keep it on this "
                             "machine (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port to serve on, 0 to pick a free one "
                             "(default: %(default)s)")
    parser.add_argument("--machine", help="render only one machine's history")
    return parser.parse_args(argv)


def load_rows(directory: Path, machine: str | None) -> tuple[list[dict], list[dict]]:
    """torchpcl's kept points, and the baselines kept beside them."""
    series, baselines = [], []
    for path in machines(directory, machine):
        series += read_rows(path / SERIES)
        baselines += read_rows(path / BASELINES)
    if not series:
        raise SystemExit(f"{directory}: no torchpcl points"
                         + (f" for machine {machine!r}" if machine else "")
                         + "; keep some with `just benchmark --keep`")
    return series, baselines


def chart_key(row: dict) -> tuple:
    return (row["machine"], row["profile"], row["profile_version"],
            row.get("group") or "", row["name"], row["device"])


def references(baselines: list[dict]) -> dict[tuple, list[dict]]:
    """Per card, each baseline backend's newest measurement."""
    newest: dict[tuple, dict] = {}
    for row in baselines:
        slot = (chart_key(row), row["backend"])
        if slot not in newest or (row["timestamp"] or "") > (newest[slot]["timestamp"] or ""):
            newest[slot] = row
    cards: dict[tuple, list[dict]] = {}
    for (key, backend), row in sorted(newest.items()):
        cards.setdefault(key, []).append({
            "backend": backend, "version": row.get("baseline_version"),
            "slot": SERIES_COLORS.get(backend, FALLBACK_SLOT),
            "median": row["median_ms"], "min": row.get("min_ms"),
            "max": row.get("max_ms"), "timestamp": row["timestamp"],
            "metrics": row.get("metrics") or {},
        })
    return cards


def build_charts(rows: list[dict], baselines: list[dict] | None = None) -> list[dict]:
    """Group points into cards, each internally comparable by construction."""
    latest: dict[tuple, int] = {}
    for row in rows:
        profile = (row["machine"], row["profile"])
        latest[profile] = max(latest.get(profile, 0), row["profile_version"])
    reference = references(baselines or [])
    charts: dict[tuple, dict] = {}
    for row in rows:
        key = chart_key(row)
        chart = charts.setdefault(key, {
            "machine": row["machine"], "suite": row["suite"],
            "profile": row["profile"], "version": row["profile_version"],
            "current": row["profile_version"] == latest[row["machine"], row["profile"]],
            "group": row.get("group"), "name": row["name"], "device": row["device"],
            "references": reference.get(key, []),
            "series": {},
        })
        chart["series"].setdefault(row["backend"], []).append(row)

    ordered = []
    for key in sorted(charts):
        chart = charts[key]
        revisions = sorted(
            {(point["timestamp"] or "", point["revision"])
             for points in chart["series"].values() for point in points})
        axis = [revision for _, revision in revisions]
        position = {revision: index for index, revision in enumerate(axis)}
        chart["revisions"] = axis
        chart["timestamps"] = [timestamp for timestamp, _ in revisions]
        chart["series"] = [
            {
                "backend": backend,
                "slot": SERIES_COLORS.get(backend, FALLBACK_SLOT),
                "points": sorted(
                    ({"x": position[point["revision"]],
                      "revision": point["revision"],
                      "timestamp": point["timestamp"],
                      "median": point["median_ms"],
                      "min": point.get("min_ms"),
                      "max": point.get("max_ms"),
                      "metrics": point.get("metrics") or {},
                      "note": point.get("note")}
                     for point in points),
                    key=lambda point: point["x"]),
            }
            for backend, points in sorted(chart["series"].items())
        ]
        ordered.append(chart)
    return ordered


def latest_movement(charts: list[dict]) -> list[dict]:
    """torchpcl's change on each card between its last two stored revisions.

    Only movement `compare.moved` accepts is called a change, so the page and
    `compare.py` can never disagree about what moved.
    """
    moves = []
    for chart in charts:
        series = next((entry for entry in chart["series"]
                       if entry["backend"] == "torchpcl"), None)
        if series is None or len(series["points"]) < 2:
            continue
        before, after = series["points"][-2], series["points"][-1]
        if not before["median"]:
            continue
        def result(point):
            return {"median": {"time_ms": point["median"]},
                    "time_ms_min": point["min"], "time_ms_max": point["max"]}

        moves.append({
            "label": f"{chart_title(chart)} · {chart['device']}",
            "case": chart_title(chart),
            "profile": chart["profile"], "version": chart["version"],
            "current": chart["current"],
            "machine": chart["machine"], "device": chart["device"],
            "before": before["median"], "after": after["median"],
            "from_revision": before["revision"], "to_revision": after["revision"],
            "ratio": after["median"] / before["median"] - 1,
            "significant": moved(result(before), result(after)),
        })
    return sorted(moves, key=lambda move: move["ratio"])


def chart_title(chart: dict) -> str:
    return "/".join(filter(None, (chart.get("group"), chart["name"])))


def page_data(rows: list[dict], baselines: list[dict] | None = None) -> dict:
    charts = build_charts(rows, baselines)
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machines": sorted({row["machine"] for row in rows}),
        "profiles": sorted({chart["profile"] for chart in charts}),
        "charts": charts,
        "movement": latest_movement(charts),
        "environments": sorted(
            {(row["machine"], row.get("platform") or "", row.get("gpu") or "",
              row["versions"].get("torch") or "", row["versions"].get("python") or "")
             for row in rows}),
    }


def render(data: dict) -> str:
    return TEMPLATE.replace("__DATA__", json.dumps(data, allow_nan=False))


def build(args) -> tuple[str, str]:
    """The rendered page, and one line describing what went into it."""
    rows, baselines = load_rows(args.history, args.machine)
    data = page_data(rows, baselines)
    return render(data), (f"{len(data['charts'])} charts from {len(rows)} points "
                          f"and {len(baselines)} baselines")


def handler_class(args):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "torchpcl-dashboard"

        def do_GET(self):
            body, status = self.page()
            self.respond(status, body)
            self.wfile.write(body)

        def do_HEAD(self):
            body, status = self.page()
            self.respond(status, body)

        def page(self) -> tuple[bytes, int]:
            # Rendered per request, not once at startup: keeping a new point
            # while the server runs should need a browser refresh, not a
            # restart. The history is a few hundred short lines, so re-reading
            # it costs less than the page transfer does.
            if self.path.split("?")[0] not in ("/", "/index.html"):
                return b"not found\n", 404
            try:
                page, summary = build(args)
            except SystemExit as error:  # an emptied or missing history
                return f"{error}\n".encode(), 503
            self.summary = summary
            return page.encode(), 200

        def respond(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type",
                             "text/html; charset=utf-8" if status == 200
                             else "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # The page carries its data inline, so a cached copy is a stale
            # copy; refreshing must show the points added since.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def log_message(self, fmt, *rest):
            if self.command == "GET" and getattr(self, "summary", None):
                print(f"served {self.summary} to {self.address_string()}",
                      flush=True)

    return Handler


def serve(args) -> int:
    page, summary = build(args)  # fail before binding if the history cannot render
    del page
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler_class(args))
    except OSError as error:
        raise SystemExit(f"cannot serve on {args.host}:{args.port}: {error}"
                         "\nPass --port N for a free one, or --port 0 to be "
                         "given one.") from error
    port = server.server_address[1]
    print(f"{args.history}: {summary}")
    print(f"  http://localhost:{port}/")
    # Flushed because this is the whole user interface, and a `just` recipe
    # piped into anything but a terminal would otherwise buffer it away.
    print("Ctrl-C to stop; the page re-renders on every reload.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out is None:
        return serve(args)
    page, summary = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"{args.out}: {summary}")
    return 0


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>torchpcl benchmark history</title>
<style>
  :root {
    color-scheme: light;
    --page:           #f9f9f7;
    --surface-1:      #fcfcfb;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --grid:           #e1e0d9;
    --axis:           #c3c2b7;
    --border:         rgba(11, 11, 11, 0.10);
    --series-1:       #2a78d6;
    --series-2:       #eb6834;
    --series-3:       #1baf7a;
    --series-4:       #eda100;
    --good:           #0ca30c;
    --critical:       #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --page:           #0d0d0d;
      --surface-1:      #1a1a19;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --grid:           #2c2c2a;
      --axis:           #383835;
      --border:         rgba(255, 255, 255, 0.10);
      --series-1:       #3987e5;
      --series-2:       #d95926;
      --series-3:       #199e70;
      --series-4:       #c98500;
    }
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 32px 16px 64px;
    background: var(--page);
    color: var(--text-primary);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: var(--text-secondary); margin: 0 0 24px; }
  .sub code { color: var(--text-secondary); }

  .filters {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end;
    margin-bottom: 24px;
  }
  .filters label { display: flex; flex-direction: column; gap: 4px;
                   font-size: 12px; color: var(--text-muted); }
  .filters select {
    font: inherit; font-size: 13px; padding: 6px 10px;
    background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 6px;
  }
  .filters .check { flex-direction: row; align-items: center; gap: 6px;
                    align-self: flex-end; padding-bottom: 7px; }

  .movement {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px; margin-bottom: 24px;
  }
  .movement h2 { font-size: 13px; font-weight: 600; margin: 0 0 10px;
                 color: var(--text-secondary); }
  .movement .columns { display: grid; gap: 12px 32px;
                       grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); }
  .movement h3 { font-size: 12px; font-weight: 600; margin: 0 0 8px;
                 text-transform: uppercase; letter-spacing: .04em; color: var(--text-primary); }
  .movement h3 span { font-weight: 400; text-transform: none; letter-spacing: 0;
                      color: var(--text-muted); margin-left: 6px; }
  .movement ul { list-style: none; margin: 0; padding: 0;
                 display: flex; flex-direction: column; gap: 6px; }
  .movement li { display: flex; gap: 10px; align-items: baseline; }
  .movement .mark { font-weight: 600; min-width: 74px;
                    font-variant-numeric: tabular-nums; }
  .movement .up { color: var(--critical); }
  .movement .down { color: var(--good); }
  .movement .case { color: var(--text-primary); }
  .movement .detail { color: var(--text-muted); font-size: 12px; }

  .overview { display: flex; flex-direction: column; gap: 18px; margin-bottom: 24px; }
  .overview .block {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px 10px; overflow-x: auto;
  }
  .overview tr.suite th { font-size: 12px; font-weight: 600; color: var(--text-secondary);
                          padding-top: 10px; }
  .overview h2 { font-size: 13px; font-weight: 600; margin: 0 0 6px;
                 color: var(--text-secondary); }
  .overview table { margin: 0; font-size: 13px; width: auto; min-width: 100%; }
  .overview thead tr:not(:last-child) th { border-bottom: none; padding-bottom: 0; }
  .overview th, .overview td { white-space: nowrap; padding: 4px 8px; }
  .overview th { font-size: 11px; text-align: left; }
  .overview th.device { text-transform: uppercase; letter-spacing: .04em; }
  .overview th.profile { color: var(--text-secondary); }
  .overview td a { color: inherit; text-decoration: none; }
  .overview td a:hover b { text-decoration: underline; }
  .overview td b { font-weight: 600; font-variant-numeric: tabular-nums; }
  .overview td { text-align: left; }
  .overview .versus { font-size: 11px; color: var(--text-muted); margin-left: 6px; }
  .overview .versus.faster { color: var(--good); }
  .overview .versus.slower { color: var(--critical); }
  .overview .mark { font-size: 11px; margin-left: 6px; color: var(--text-muted);
                    font-variant-numeric: tabular-nums; }
  .overview .mark.up { color: var(--critical); }
  .overview .mark.down { color: var(--good); }

  details.device-section { margin: 0 0 12px; border-top: 1px solid var(--border); }
  details.device-section > summary {
    font-size: 22px; font-weight: 600; color: var(--text-primary);
    padding: 14px 0; list-style-position: inside;
  }
  details.device-section[open] > summary { padding-bottom: 18px; }
  details.device-section > summary .count { font-size: 13px; font-weight: 400;
                                            color: var(--text-muted); margin-left: 10px; }

  .grid { display: grid; gap: 18px;
          grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }
  .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 16px 12px; scroll-margin-top: 16px;
  }
  /* The card a matrix cell jumped to. */
  .card:target { outline: 2px solid var(--series-1); outline-offset: 2px;
                 animation: arrive 1.2s ease-out; }
  @keyframes arrive {
    from { box-shadow: 0 0 0 8px color-mix(in srgb, var(--series-1) 35%, transparent); }
    to   { box-shadow: 0 0 0 0 transparent; }
  }
  @media (prefers-reduced-motion: reduce) { .card:target { animation: none; } }
  .card h3 { font-size: 14px; font-weight: 600; margin: 0; }
  /* Without an explicit ratio the scaled SVG does not contribute its height to
     the card, and the legend and table below it spill past the card border. */
  .card svg { display: block; width: 100%; height: auto; aspect-ratio: 340 / 170; }
  .card .meta { font-size: 12px; color: var(--text-muted); margin: 2px 0 10px; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 0;
            font-size: 12px; color: var(--text-secondary); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .legend i { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }
  .legend i.dashed, .tooltip i.dashed { height: 0; background: none !important;
                                        border-top: 2px dashed currentColor; }

  details { margin-top: 10px; }
  summary { cursor: pointer; font-size: 12px; color: var(--text-muted); }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12px; }
  th, td { text-align: right; padding: 3px 6px; border-bottom: 1px solid var(--grid);
           font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
  th { color: var(--text-muted); font-weight: 500; }

  .tooltip {
    position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 12px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, .18); z-index: 10; max-width: 260px;
  }
  .tooltip .head { color: var(--text-muted); margin-bottom: 4px; }
  .tooltip .row { display: flex; align-items: baseline; gap: 6px; }
  .tooltip .row b { font-weight: 600; font-variant-numeric: tabular-nums; }
  .tooltip .row span { color: var(--text-secondary); }
  .tooltip i { width: 10px; height: 2px; border-radius: 1px; display: inline-block; }

  .empty { color: var(--text-muted); padding: 32px 0; }
  footer { margin-top: 32px; color: var(--text-muted); font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>torchpcl benchmark history</h1>
  <p class="sub" id="subtitle"></p>

  <div class="filters">
    <label>Machine <select id="machine"></select></label>
    <label>Profile <select id="profile"></select></label>
    <label class="check"><input type="checkbox" id="retired"> superseded profile versions</label>
  </div>

  <section class="overview" id="overview" hidden></section>

  <section class="movement" id="movement" hidden>
    <h2 id="movement-title"></h2>
    <div class="columns" id="movement-columns"></div>
  </section>

  <div id="sections"></div>
  <p class="empty" id="empty" hidden>No cards match these filters.</p>

  <footer id="environments"></footer>
</div>
<div class="tooltip" id="tooltip" role="status" aria-live="polite"></div>

<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const PLOT = { width: 340, height: 170, left: 46, right: 16, top: 12, bottom: 34 };

const el = (tag, props = {}, kids = []) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  kids.forEach((kid) => node.appendChild(kid));
  return node;
};
const svgEl = (tag, props = {}) => {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(props)) node.setAttribute(k, v);
  return node;
};
const color = (slot) => `var(--series-${slot})`;
const fmtMs = (value) =>
  value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
const shortRev = (revision) => revision || "";

// Clean axis steps: 1, 2 or 5 times a power of ten.
function ticks(low, high, count = 4) {
  if (!(high > low)) return [low];
  const raw = (high - low) / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) || magnitude * 10;
  const out = [];
  for (let v = Math.ceil(low / step) * step; v <= high + step / 1000; v += step) out.push(v);
  return out;
}

function chartBounds(chart) {
  let low = Infinity;
  let high = -Infinity;
  for (const series of [...chart.series, { points: chart.references }]) {
    for (const point of series.points) {
      for (const value of [point.median, point.min, point.max]) {
        if (typeof value === "number") {
          low = Math.min(low, value);
          high = Math.max(high, value);
        }
      }
    }
  }
  if (!isFinite(low)) return [0, 1];
  if (low === high) return [low * 0.95, high * 1.05 || 1];
  const pad = (high - low) * 0.12;
  return [Math.max(0, low - pad), high + pad];
}

function drawChart(chart) {
  const { width, height, left, right, top, bottom } = PLOT;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const [low, high] = chartBounds(chart);
  const count = chart.revisions.length;
  const x = (index) => left + (count < 2 ? plotWidth / 2 : (index / (count - 1)) * plotWidth);
  const y = (value) => top + plotHeight - ((value - low) / (high - low)) * plotHeight;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img", "aria-label": `${title(chart)} on ${chart.device}, milliseconds by revision`,
  });

  for (const value of ticks(low, high)) {
    svg.appendChild(svgEl("line", {
      x1: left, x2: width - right, y1: y(value), y2: y(value),
      stroke: "var(--grid)", "stroke-width": 1,
    }));
    const label = svgEl("text", {
      x: left - 8, y: y(value) + 4, "text-anchor": "end",
      fill: "var(--text-muted)", "font-size": 10,
      style: "font-variant-numeric: tabular-nums",
    });
    label.textContent = fmtMs(value);
    svg.appendChild(label);
  }
  svg.appendChild(svgEl("line", {
    x1: left, x2: width - right, y1: top + plotHeight, y2: top + plotHeight,
    stroke: "var(--axis)", "stroke-width": 1,
  }));

  // X labels: as many revisions as fit without touching. The last is always
  // shown -- it is the one the reader came for -- so a neighbour too close to
  // it is dropped rather than allowed to overprint it.
  const MIN_LABEL_GAP = 56;
  const shown = [];
  for (let index = 0; index < count; index += 1) {
    if (!shown.length || x(index) - x(shown[shown.length - 1]) >= MIN_LABEL_GAP) {
      shown.push(index);
    }
  }
  while (shown.length > 1
         && x(count - 1) - x(shown[shown.length - 1]) < MIN_LABEL_GAP
         && shown[shown.length - 1] !== count - 1) {
    shown.pop();
  }
  if (shown[shown.length - 1] !== count - 1) shown.push(count - 1);
  chart.revisions.forEach((revision, index) => {
    if (!shown.includes(index)) return;
    const anchor = index === 0 ? "start" : index === count - 1 ? "end" : "middle";
    const label = svgEl("text", {
      x: x(index), y: height - bottom + 16, "text-anchor": count < 2 ? "middle" : anchor,
      fill: "var(--text-muted)", "font-size": 10,
    });
    label.textContent = shortRev(revision);
    svg.appendChild(label);
  });

  // Baselines: one measurement per library version, not a series, so a
  // dashed level across the whole plot with its measured range behind it.
  for (const ref of chart.references) {
    const stroke = color(ref.slot);
    if (typeof ref.min === "number" && typeof ref.max === "number") {
      svg.appendChild(svgEl("rect", {
        x: left, width: plotWidth, y: y(ref.max),
        height: Math.max(0.5, y(ref.min) - y(ref.max)),
        fill: stroke, "fill-opacity": 0.08,
      }));
    }
    svg.appendChild(svgEl("line", {
      x1: left, x2: width - right, y1: y(ref.median), y2: y(ref.median),
      stroke, "stroke-width": 1.5, "stroke-dasharray": "5 4",
    }));
  }

  for (const series of chart.series) {
    const stroke = color(series.slot);
    const band = series.points.filter((p) => typeof p.min === "number" && typeof p.max === "number");
    if (band.length > 1) {
      const upper = band.map((p) => `${x(p.x)},${y(p.max)}`);
      const lower = band.slice().reverse().map((p) => `${x(p.x)},${y(p.min)}`);
      svg.appendChild(svgEl("polygon", {
        points: [...upper, ...lower].join(" "), fill: stroke,
        "fill-opacity": 0.10, stroke: "none",
      }));
    }
    if (series.points.length > 1) {
      svg.appendChild(svgEl("polyline", {
        points: series.points.map((p) => `${x(p.x)},${y(p.median)}`).join(" "),
        fill: "none", stroke, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-linecap": "round",
      }));
    }
    for (const point of series.points) {
      svg.appendChild(svgEl("circle", {
        cx: x(point.x), cy: y(point.median), r: 4, fill: stroke,
        stroke: "var(--surface-1)", "stroke-width": 2,
      }));
    }
  }

  // Direct end-labels: the relief for the light-mode contrast warning, and what
  // keeps a value readable without hovering. torchpcl is labelled first because
  // it is the subject; a baseline label that would collide with one already
  // placed is dropped rather than nudged off its line -- the legend, tooltip
  // and table still carry it.
  const placed = [];
  const labelled = [...chart.series, ...chart.references.map((ref) => ({ points: [ref] }))];
  for (const series of labelled) {
    const last = series.points[series.points.length - 1];
    if (!last) continue;
    const at = y(last.median) - 7;
    if (placed.some((other) => Math.abs(other - at) < 11)) continue;
    placed.push(at);
    const label = svgEl("text", {
      x: width - right, y: at, "text-anchor": "end",
      fill: "var(--text-secondary)", "font-size": 10,
      style: "font-variant-numeric: tabular-nums",
    });
    label.textContent = fmtMs(last.median);
    svg.appendChild(label);
  }

  const crosshair = svgEl("line", {
    y1: top, y2: top + plotHeight, stroke: "var(--axis)",
    "stroke-width": 1, opacity: 0,
  });
  svg.appendChild(crosshair);
  const surface = svgEl("rect", {
    x: left, y: top, width: plotWidth, height: plotHeight,
    fill: "transparent", style: "cursor: crosshair",
  });
  svg.appendChild(surface);
  attachHover(svg, surface, crosshair, chart, x);
  return svg;
}

const tooltip = document.getElementById("tooltip");

function attachHover(svg, surface, crosshair, chart, x) {
  const nearest = (event) => {
    const box = svg.getBoundingClientRect();
    const local = ((event.clientX - box.left) / box.width) * PLOT.width;
    let best = 0;
    let bestDistance = Infinity;
    chart.revisions.forEach((_, index) => {
      const distance = Math.abs(x(index) - local);
      if (distance < bestDistance) { bestDistance = distance; best = index; }
    });
    return best;
  };
  const show = (event) => {
    const index = nearest(event);
    crosshair.setAttribute("x1", x(index));
    crosshair.setAttribute("x2", x(index));
    crosshair.setAttribute("opacity", 1);
    renderTooltip(chart, index, event.clientX, event.clientY);
  };
  surface.addEventListener("pointermove", show);
  surface.addEventListener("pointerdown", show);
  surface.addEventListener("pointerleave", () => {
    crosshair.setAttribute("opacity", 0);
    tooltip.style.opacity = 0;
  });
}

function renderTooltip(chart, index, clientX, clientY) {
  tooltip.textContent = "";
  const revision = chart.revisions[index];
  const when = (chart.timestamps[index] || "").slice(0, 16).replace("T", " ");
  tooltip.appendChild(el("div", { class: "head", text: `${shortRev(revision)} · ${when}` }));
  for (const series of chart.series) {
    const point = series.points.find((p) => p.x === index);
    if (!point) continue;
    const key = el("i");
    key.style.background = color(series.slot);
    const spread = (typeof point.min === "number" && typeof point.max === "number")
      ? ` (${fmtMs(point.min)}–${fmtMs(point.max)})` : "";
    const row = el("div", { class: "row" }, [
      key,
      el("b", { text: `${fmtMs(point.median)} ms${spread}` }),
      el("span", { text: series.backend }),
    ]);
    tooltip.appendChild(row);
    for (const [name, value] of Object.entries(point.metrics)) {
      if (typeof value !== "number") continue;
      tooltip.appendChild(el("div", { class: "row" }, [
        el("span", { text: `${name} ${value.toPrecision(4)}` }),
      ]));
    }
  }
  for (const ref of chart.references) {
    const key = el("i", { class: "dashed" });
    key.style.color = color(ref.slot);
    tooltip.appendChild(el("div", { class: "row" }, [
      key, el("b", { text: `${fmtMs(ref.median)} ms` }), el("span", { text: refLabel(ref) }),
    ]));
  }
  tooltip.style.opacity = 1;
  const box = tooltip.getBoundingClientRect();
  const x = Math.min(clientX + 14, window.innerWidth - box.width - 8);
  const y = Math.min(clientY + 14, window.innerHeight - box.height - 8);
  tooltip.style.left = `${Math.max(8, x)}px`;
  tooltip.style.top = `${Math.max(8, y)}px`;
}

function drawTable(chart) {
  const head = el("tr", {}, [el("th", { text: "revision" })]);
  chart.series.forEach((series) => head.appendChild(el("th", { text: `${series.backend} ms` })));
  const body = chart.revisions.map((revision, index) => {
    const row = el("tr", {}, [el("td", { text: shortRev(revision) })]);
    chart.series.forEach((series) => {
      const point = series.points.find((p) => p.x === index);
      row.appendChild(el("td", { text: point ? fmtMs(point.median) : "–" }));
    });
    return row;
  });
  chart.references.forEach((ref) => {
    const row = el("tr", {}, [el("td", { text: refLabel(ref) })]);
    chart.series.forEach((_, index) => row.appendChild(
      el("td", { text: index === 0 ? fmtMs(ref.median) : "" })));
    body.push(row);
  });
  return el("details", {}, [
    el("summary", { text: "Data table" }),
    el("table", {}, [el("thead", {}, [head]), el("tbody", {}, body)]),
  ]);
}

const title = (chart) => [chart.group, chart.name].filter(Boolean).join("/");
const refLabel = (ref) => `${ref.backend} ${ref.version || ""}`.trim();

// Cases are keyed without their device, so the overview can put a case's
// devices side by side and a matrix cell can point at its card.
const caseKey = (chart) =>
  [chart.machine, chart.profile, chart.version, chart.group || "", chart.name].join("|");
const cardId = (chart) => `card-${caseKey(chart)}|${chart.device}`.replace(/[^\w.-]/g, "_");

// CUDA first: it is what torchpcl is for. Any other device follows the two.
const DEVICE_ORDER = ["cuda", "cpu"];
const deviceRank = (device) => {
  const rank = DEVICE_ORDER.indexOf(device);
  return rank < 0 ? DEVICE_ORDER.length : rank;
};
const byDevice = (a, b) => deviceRank(a) - deviceRank(b) || a.localeCompare(b);

// torchpcl's newest point against the fastest baseline on the same card.
// Coloured only when the measured ranges are disjoint -- the same test the
// movement banner applies -- so overlapping noise never reads as a win.
function versus(chart) {
  const series = chart.series.find((entry) => entry.backend === "torchpcl");
  const last = series && series.points[series.points.length - 1];
  if (!last || !chart.references.length) return null;
  const best = chart.references.reduce((a, b) => (b.median < a.median ? b : a));
  const faster = best.median >= last.median;
  const factor = faster ? best.median / last.median : last.median / best.median;
  const known = [last.min, last.max, best.min, best.max].every((v) => typeof v === "number");
  const disjoint = known && (last.max < best.min || best.max < last.min);
  return {
    text: `${factor.toFixed(factor >= 10 ? 0 : 1)}× ${faster ? "faster" : "slower"} than ${best.backend}`,
    title: `${fmtMs(best.median)} ms, ${refLabel(best)}`,
    tone: disjoint ? (faster ? "faster" : "slower") : "",
  };
}

// A suite measured under several profiles gets a block of its own, with its
// cases as rows and each profile's devices as columns, so a case measured on
// two inputs reads across one line. Suites with a single profile share one
// block whose columns are just the devices.
function drawOverview(charts, moves) {
  const section = document.getElementById("overview");
  section.textContent = "";
  section.hidden = !charts.length;
  if (!charts.length) return;

  const machines = new Set(charts.map((chart) => chart.machine));
  const moved = new Map(moves.map(
    (m) => [`${m.machine}|${m.profile}|${m.version}|${m.label}`, m]));
  const profilesOf = new Map();
  for (const chart of charts) {
    const suite = `${chart.machine}|${chart.suite}`;
    if (!profilesOf.has(suite)) profilesOf.set(suite, new Set());
    profilesOf.get(suite).add(`${chart.profile}|${chart.version}`);
  }
  const blocks = new Map();
  for (const chart of charts) {
    const shared = profilesOf.get(`${chart.machine}|${chart.suite}`).size === 1;
    const blockKey = `${chart.machine}|${shared ? "" : chart.suite}`;
    if (!blocks.has(blockKey)) {
      const name = shared ? "" : chart.suite;
      blocks.set(blockKey, {
        title: [name, machines.size > 1 ? chart.machine : ""].filter(Boolean).join(" · "),
        shared, profiles: new Map(), rows: new Map() });
    }
    const block = blocks.get(blockKey);
    const profileKey = shared ? "" : `${chart.profile}|${chart.version}`;
    if (!block.profiles.has(profileKey)) {
      const short = chart.profile.startsWith(`${chart.suite}-`)
        ? chart.profile.slice(chart.suite.length + 1) : chart.profile;
      block.profiles.set(profileKey, {
        label: chart.current ? short : `${short} v${chart.version}`, devices: new Set() });
    }
    block.profiles.get(profileKey).devices.add(chart.device);
    const rowKey = `${chart.suite}|${title(chart)}`;
    if (!block.rows.has(rowKey)) {
      block.rows.set(rowKey, { suite: chart.suite, name: title(chart), cells: new Map() });
    }
    block.rows.get(rowKey).cells.set(`${profileKey}|${chart.device}`, chart);
  }

  for (const block of blocks.values()) {
    const columns = [];
    const profileRow = el("tr", {}, [el("th")]);
    const deviceRow = el("tr", {}, [el("th", { text: "case" })]);
    for (const [profileKey, profile] of block.profiles) {
      const devices = [...profile.devices].sort(byDevice);
      profileRow.appendChild(el("th", {
        class: "profile", colspan: devices.length, text: profile.label }));
      devices.forEach((device) => {
        columns.push(`${profileKey}|${device}`);
        deviceRow.appendChild(el("th", { class: "device", text: device }));
      });
    }
    const head = block.profiles.size > 1 ? [profileRow, deviceRow] : [deviceRow];
    const body = [];
    let suite = null;
    for (const row of block.rows.values()) {
      if (block.shared && row.suite !== suite) {
        suite = row.suite;
        body.push(el("tr", { class: "suite" }, [
          el("th", { colspan: columns.length + 1, text: suite })]));
      }
      const tr = el("tr", {}, [el("td", { text: row.name })]);
      columns.forEach((column) => tr.appendChild(drawCell(row.cells.get(column), moved)));
      body.push(tr);
    }
    const kids = block.title ? [el("h2", { text: block.title })] : [];
    kids.push(el("table", {}, [el("thead", {}, head), el("tbody", {}, body)]));
    section.appendChild(el("div", { class: "block" }, kids));
  }
}

// Opens the section holding a card so an anchor to it has somewhere to land.
function reveal(id) {
  const card = document.getElementById(id);
  const section = card && card.closest("details");
  if (section) section.open = true;
  return card;
}

function drawCell(chart, moved) {
  const series = chart && chart.series.find((entry) => entry.backend === "torchpcl");
  const last = series && series.points[series.points.length - 1];
  if (!last) return el("td", { text: "–" });
  const link = el("a", { href: `#${cardId(chart)}` }, [
    el("b", { text: `${fmtMs(last.median)} ms` })]);
  const move = moved.get(
    `${chart.machine}|${chart.profile}|${chart.version}|${title(chart)} · ${chart.device}`);
  // Change since torchpcl's previous stored revision. Coloured only when
  // `compare.moved` accepts it; within the measured spread it stays muted.
  if (move) {
    const tone = move.significant ? (move.ratio > 0 ? "up" : "down") : "";
    link.appendChild(el("span", {
      class: `mark ${tone}`,
      text: `${move.ratio > 0 ? "▲" : "▼"}${Math.abs(move.ratio * 100).toFixed(1)}%`,
      title: `${fmtMs(move.before)} → ${fmtMs(move.after)} ms, `
             + `${shortRev(move.from_revision)} → ${shortRev(move.to_revision)}`,
    }));
  }
  const against = versus(chart);
  if (against) {
    link.appendChild(el("span", {
      class: `versus ${against.tone}`, text: against.text, title: against.title }));
  }
  link.addEventListener("click", () => reveal(cardId(chart)));
  return el("td", {}, [link]);
}

// Collapsed on load; a filter change redraws the cards but keeps whichever
// sections the reader has opened since.
function drawSections(charts) {
  const container = document.getElementById("sections");
  const open = new Set([...container.querySelectorAll("details[open]")]
    .map((section) => section.dataset.device));
  container.textContent = "";
  const devices = [...new Set(charts.map((chart) => chart.device))].sort(byDevice);
  for (const device of devices) {
    const cards = charts.filter((chart) => chart.device === device);
    const grid = el("div", { class: "grid" });
    cards.forEach((chart) => grid.appendChild(drawCard(chart)));
    const section = el("details", {
      class: "device-section", id: `section-${device}`, "data-device": device }, [
      el("summary", {}, [
        document.createTextNode(device.toUpperCase()),
        el("span", { class: "count", text: `${cards.length} cases` }),
      ]),
      grid,
    ]);
    section.open = open.has(device);
    container.appendChild(section);
  }
}

function drawCard(chart) {
  const version = chart.current ? "" : ` · superseded v${chart.version}`;
  const card = el("div", { class: "card", id: cardId(chart) }, [
    el("h3", { text: `${title(chart)} · ${chart.device}` }),
    el("p", { class: "meta", text: `${chart.profile}${version}` }),
  ]);
  card.appendChild(drawChart(chart));
  if (chart.series.length + chart.references.length > 1) {
    const legend = el("div", { class: "legend" });
    chart.series.forEach((series) => {
      const key = el("i");
      key.style.background = color(series.slot);
      legend.appendChild(el("span", {}, [key, el("span", { text: series.backend })]));
    });
    chart.references.forEach((ref) => {
      const key = el("i", { class: "dashed" });
      key.style.color = color(ref.slot);
      legend.appendChild(el("span", {}, [key, el("span", { text: refLabel(ref) })]));
    });
    card.appendChild(legend);
  }
  card.appendChild(drawTable(chart));
  return card;
}

// One column per device, CUDA first, so a change on one device is read
// beside the other rather than interleaved with it.
function drawMovement(moves) {
  const section = document.getElementById("movement");
  const columns = document.getElementById("movement-columns");
  columns.textContent = "";
  section.hidden = !moves.length;
  if (!moves.length) return;
  document.getElementById("movement-title").textContent =
    "torchpcl, newest stored revision vs the one before it";
  const devices = [...new Set(moves.map((m) => m.device))].sort(byDevice);
  for (const device of devices) {
    const real = moves.filter((m) => m.device === device && m.significant);
    const slower = real.filter((m) => m.ratio > 0).length;
    const summary = real.length
      ? `${slower} slower, ${real.length - slower} faster`
      : "nothing moved beyond its measured spread";
    const list = el("ul");
    for (const move of real) {
      list.appendChild(el("li", {}, [
        el("span", {
          class: `mark ${move.ratio > 0 ? "up" : "down"}`,
          text: `${move.ratio > 0 ? "▲" : "▼"} ${(move.ratio * 100).toFixed(1)}%`,
        }),
        el("span", { class: "case", text: move.case }),
        el("span", {
          class: "detail",
          text: `${fmtMs(move.before)} → ${fmtMs(move.after)} ms · ${move.profile} · `
                + `${shortRev(move.from_revision)} → ${shortRev(move.to_revision)}`,
        }),
      ]));
    }
    columns.appendChild(el("div", {}, [
      el("h3", {}, [document.createTextNode(device), el("span", { text: summary })]),
      list,
    ]));
  }
}

function fill(select, values, allLabel) {
  select.textContent = "";
  select.appendChild(el("option", { value: "", text: allLabel }));
  values.forEach((value) => select.appendChild(el("option", { value, text: value })));
}

function apply() {
  const machine = document.getElementById("machine").value;
  const profile = document.getElementById("profile").value;
  const retired = document.getElementById("retired").checked;
  const matches = (item) =>
    (!machine || item.machine === machine)
    && (!profile || item.profile === profile)
    && (retired || item.current);

  const charts = DATA.charts.filter(matches);
  const moves = DATA.movement.filter(matches);
  drawOverview(charts, moves);
  drawSections(charts);
  document.getElementById("empty").hidden = charts.length > 0;
  drawMovement(moves);
}

function init() {
  const points = DATA.charts.reduce(
    (total, chart) => total + chart.series.reduce((n, s) => n + s.points.length, 0), 0);
  document.getElementById("subtitle").textContent =
    `${points} points · ${DATA.charts.length} cases · generated ${DATA.generated.replace("T", " ")}`;

  fill(document.getElementById("machine"), DATA.machines, "All machines");
  fill(document.getElementById("profile"), DATA.profiles, "All profiles");
  ["machine", "profile", "retired"].forEach((id) =>
    document.getElementById(id).addEventListener("change", apply));

  const footer = document.getElementById("environments");
  DATA.environments.forEach(([machine, platform, gpu, torch, python]) => {
    footer.appendChild(el("div", {
      text: [machine, platform, gpu || "no GPU recorded", `torch ${torch}`,
             `python ${python}`].filter(Boolean).join(" · "),
    }));
  });
  apply();
  // A shared link to a card: its section starts collapsed like the others,
  // so open it before the browser's own jump would have landed.
  const linked = location.hash && reveal(decodeURIComponent(location.hash.slice(1)));
  if (linked) linked.scrollIntoView();
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
