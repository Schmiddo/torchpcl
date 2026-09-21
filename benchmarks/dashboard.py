"""Render the saved benchmark history as one self-contained HTML page.

`history.py` keeps the points; this turns them into charts. By default it
serves them over HTTP, re-rendering on each request so a newly kept point needs
only a reload -- and so the page can be read from another computer. With
`--out` it writes the same page to a file instead: the data is embedded in it,
so that copy needs no server, no network and no dependencies and can be
attached to a message and still work.

One card per measured case, at most three lines on it: torchpcl and whichever
baselines were run for the same case. Each line carries its measured min/max as
a band, because a median alone cannot distinguish a regression from a busy
machine. Cards are keyed on the configuration digest, so a parameter change
starts a new card rather than bending an existing line.
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

from benchmarks.history import DEFAULT_HISTORY, read_configs, read_history

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


def load_rows(directory: Path, machine: str | None) -> list[dict]:
    if not directory.is_dir():
        raise SystemExit(f"{directory}: no history yet; add a report with "
                         "`python benchmarks/history.py add runs/<report>.json`")
    rows = [row for path in sorted(directory.glob("*.jsonl"))
            for row in read_history(path)]
    if machine:
        rows = [row for row in rows if row["machine"] == machine]
    if not rows:
        raise SystemExit(f"{directory}: no points"
                         + (f" for machine {machine!r}" if machine else ""))
    return rows


def chart_key(row: dict) -> tuple:
    return (row["machine"], row["suite"], row.get("variant"),
            row["config_digest"], row["name"], row["device"])


def distinguishing_fields(rows: list[dict], configs: dict) -> dict[tuple, list[str]]:
    """Per (suite, variant), the config fields that actually differ between digests.

    A digest is opaque, so a card has to say what it stands for. Naming every
    parameter would bury the one that changed; naming the ones that differ is
    the same rule `compare.py` applies when it prints config differences.
    """
    groups: dict[tuple, set[str]] = {}
    for row in rows:
        groups.setdefault((row["suite"], row.get("variant")), set()).add(row["config_digest"])
    fields = {}
    for group, digests in groups.items():
        names = {name for digest in digests for name in configs.get(digest, {})}
        differing = sorted(
            name for name in names
            if len({json.dumps(configs.get(digest, {}).get(name), default=str)
                    for digest in digests}) > 1)
        fields[group] = differing
    return fields


def config_label(row: dict, configs: dict, fields: dict[tuple, list[str]]) -> str:
    """A short, honest description of the card's configuration."""
    config = configs.get(row["config_digest"], {})
    differing = fields.get((row["suite"], row.get("variant")), [])
    shown = differing or [name for name in ("threads", "dtype", "voxel_size")
                          if name in config]
    parts = [f"{name}={config[name]}" for name in shown if name in config]
    return "  ".join(parts) or f"config {row['config_digest']}"


def build_charts(rows: list[dict], configs: dict) -> list[dict]:
    """Group points into cards, each internally comparable by construction."""
    fields = distinguishing_fields(rows, configs)
    charts: dict[tuple, dict] = {}
    for row in rows:
        key = chart_key(row)
        chart = charts.setdefault(key, {
            "machine": row["machine"], "suite": row["suite"],
            "variant": row.get("variant"), "digest": row["config_digest"],
            "name": row["name"], "device": row["device"],
            "config": config_label(row, configs, fields),
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

    Only movement whose measured ranges do not overlap is called a change --
    the same rule `compare.py` uses, for the same reason: anything inside the
    spread is indistinguishable from clock drift.
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
        ratio = after["median"] / before["median"] - 1
        separated = (before["max"] is not None and after["min"] is not None
                     and (before["max"] < after["min"] or after["max"] < before["min"]))
        moves.append({
            "label": f"{chart['name']} · {chart['device']}",
            "suite": chart["suite"], "machine": chart["machine"],
            "device": chart["device"],
            "before": before["median"], "after": after["median"],
            "from_revision": before["revision"], "to_revision": after["revision"],
            "ratio": ratio, "separated": separated,
        })
    return sorted(moves, key=lambda move: move["ratio"])


def page_data(rows: list[dict], configs: dict) -> dict:
    charts = build_charts(rows, configs)
    machines = sorted({row["machine"] for row in rows})
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machines": machines,
        "suites": sorted({chart["suite"] for chart in charts}),
        "devices": sorted({chart["device"] for chart in charts}),
        "charts": charts,
        "movement": latest_movement(charts),
        "environments": sorted(
            {(row["machine"], row.get("platform") or "", row.get("gpu") or "",
              row.get("torch") or "", row.get("python") or "")
             for row in rows}),
    }


def render(data: dict) -> str:
    return TEMPLATE.replace("__DATA__", json.dumps(data, allow_nan=False))


def build(args) -> tuple[str, str]:
    """The rendered page, and one line describing what went into it."""
    rows = load_rows(args.history, args.machine)
    data = page_data(rows, read_configs(args.history))
    return render(data), f"{len(data['charts'])} charts from {len(rows)} points"


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
    :root:where(:not([data-theme="light"])) {
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
  :root[data-theme="dark"] {
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

  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 32px 16px 64px;
    background: var(--page);
    color: var(--text-primary);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: var(--text-secondary); margin: 0 0 24px; }
  .sub code { color: var(--text-secondary); }

  .filters {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end;
    margin-bottom: 24px;
  }
  .filters label { display: flex; flex-direction: column; gap: 4px;
                   font-size: 12px; color: var(--text-muted); }
  .filters select, .filters button {
    font: inherit; font-size: 13px; padding: 6px 10px;
    background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 6px;
  }
  .filters button { cursor: pointer; align-self: flex-end; }

  .movement {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px; margin-bottom: 24px;
  }
  .movement h2 { font-size: 13px; font-weight: 600; margin: 0 0 10px;
                 color: var(--text-secondary); }
  .movement ul { list-style: none; margin: 0; padding: 0;
                 display: flex; flex-direction: column; gap: 6px; }
  .movement li { display: flex; gap: 10px; align-items: baseline; }
  .movement .mark { font-weight: 600; min-width: 74px;
                    font-variant-numeric: tabular-nums; }
  .movement .up { color: var(--critical); }
  .movement .down { color: var(--good); }
  .movement .case { color: var(--text-primary); }
  .movement .detail { color: var(--text-muted); font-size: 12px; }

  .grid { display: grid; gap: 18px;
          grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }
  .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 16px 12px;
  }
  .card h3 { font-size: 14px; font-weight: 600; margin: 0; }
  /* Without an explicit ratio the scaled SVG does not contribute its height to
     the card, and the legend and table below it spill past the card border. */
  .card svg { display: block; width: 100%; height: auto; aspect-ratio: 340 / 170; }
  .card .meta { font-size: 12px; color: var(--text-muted); margin: 2px 0 10px; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 0;
            font-size: 12px; color: var(--text-secondary); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .legend i { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }

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
    <label>Suite <select id="suite"></select></label>
    <label>Device <select id="device"></select></label>
    <button id="theme" type="button">Toggle theme</button>
  </div>

  <section class="movement" id="movement" hidden>
    <h2 id="movement-title"></h2>
    <ul id="movement-list"></ul>
  </section>

  <div class="grid" id="grid"></div>
  <p class="empty" id="empty" hidden>No cards match these filters.</p>

  <footer id="environments"></footer>
</div>
<div class="tooltip" id="tooltip" role="status" aria-live="polite"></div>

<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const PLOT = { width: 340, height: 170, left: 46, right: 16, top: 12, bottom: 34 };
const NOISE = 0.02;

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
const shortRev = (revision) => (revision || "").replace("-dirty", "*");

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
  for (const series of chart.series) {
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
    role: "img", "aria-label": `${chart.name} on ${chart.device}, milliseconds by revision`,
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
  const labelled = chart.series.slice().sort(
    (a, b) => (a.backend === "torchpcl" ? -1 : 0) - (b.backend === "torchpcl" ? -1 : 0));
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
  return el("details", {}, [
    el("summary", { text: "Data table" }),
    el("table", {}, [el("thead", {}, [head]), el("tbody", {}, body)]),
  ]);
}

function drawCard(chart) {
  const title = `${chart.name} · ${chart.device}`;
  const suite = chart.variant ? `${chart.suite} (${chart.variant})` : chart.suite;
  const card = el("div", { class: "card" }, [
    el("h3", { text: title }),
    el("p", { class: "meta", text: `${suite} · ${chart.config}` }),
  ]);
  card.appendChild(drawChart(chart));
  if (chart.series.length > 1) {
    const legend = el("div", { class: "legend" });
    chart.series.forEach((series) => {
      const key = el("i");
      key.style.background = color(series.slot);
      legend.appendChild(el("span", {}, [key, el("span", { text: series.backend })]));
    });
    card.appendChild(legend);
  }
  card.appendChild(drawTable(chart));
  return card;
}

function drawMovement(moves) {
  const section = document.getElementById("movement");
  const list = document.getElementById("movement-list");
  const title = document.getElementById("movement-title");
  list.textContent = "";
  const real = moves.filter((m) => Math.abs(m.ratio) > NOISE && m.separated);
  if (!moves.length) { section.hidden = true; return; }
  section.hidden = false;
  title.textContent = real.length
    ? `torchpcl, newest stored revision vs the one before it: `
      + `${real.filter((m) => m.ratio > 0).length} slower, `
      + `${real.filter((m) => m.ratio < 0).length} faster`
    : "torchpcl, newest stored revision vs the one before it: "
      + "nothing moved beyond its measured spread";
  for (const move of real) {
    const mark = el("span", {
      class: `mark ${move.ratio > 0 ? "up" : "down"}`,
      text: `${move.ratio > 0 ? "▲" : "▼"} ${(move.ratio * 100).toFixed(1)}%`,
    });
    list.appendChild(el("li", {}, [
      mark,
      el("span", { class: "case", text: move.label }),
      el("span", {
        class: "detail",
        text: `${fmtMs(move.before)} → ${fmtMs(move.after)} ms · ${move.suite} · `
              + `${shortRev(move.from_revision)} → ${shortRev(move.to_revision)}`,
      }),
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
  const suite = document.getElementById("suite").value;
  const device = document.getElementById("device").value;
  const matches = (item) =>
    (!machine || item.machine === machine)
    && (!suite || item.suite === suite)
    && (!device || item.device === device);

  const grid = document.getElementById("grid");
  grid.textContent = "";
  const charts = DATA.charts.filter(matches);
  charts.forEach((chart) => grid.appendChild(drawCard(chart)));
  document.getElementById("empty").hidden = charts.length > 0;
  drawMovement(DATA.movement.filter(matches));
}

function init() {
  const points = DATA.charts.reduce(
    (total, chart) => total + chart.series.reduce((n, s) => n + s.points.length, 0), 0);
  document.getElementById("subtitle").textContent =
    `${points} points · ${DATA.charts.length} cases · generated ${DATA.generated.replace("T", " ")}`;

  fill(document.getElementById("machine"), DATA.machines, "All machines");
  fill(document.getElementById("suite"), DATA.suites, "All suites");
  fill(document.getElementById("device"), DATA.devices, "All devices");
  ["machine", "suite", "device"].forEach((id) =>
    document.getElementById(id).addEventListener("change", apply));

  document.getElementById("theme").addEventListener("click", () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
  });

  const footer = document.getElementById("environments");
  if (DATA.charts.some((chart) => chart.revisions.some((r) => (r || "").endsWith("-dirty")))) {
    footer.appendChild(el("div", {
      text: "* marks a revision measured with uncommitted changes: the point "
            + "cannot be attributed to that commit alone.",
    }));
  }
  DATA.environments.forEach(([machine, platform, gpu, torch, python]) => {
    footer.appendChild(el("div", {
      text: [machine, platform, gpu || "no GPU recorded", `torch ${torch}`,
             `python ${python}`].filter(Boolean).join(" · "),
    }));
  });
  apply();
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
