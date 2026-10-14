# Benchmarks

Track torchpcl performance per revision, against Open3D and small-gicp as
reference baselines. Run commands from the repository root.

```bash
uv sync --group benchmark
just benchmark                  # every routine profile on this revision
just benchmark --keep           # ... and keep it in the history (committed tree only)
just benchmark-baselines --keep # Open3D/small-gicp reference values, per library version
just benchmark-backfill         # keep each unpushed commit (or a range A..B), built in a worktree
just benchmark-dashboard        # plot the history on :8770
just benchmark-run operations --op knn --k 16   # ad hoc: any suite, any parameters
```

There are four suites -- `operations`, `sampling`, `registration` and
`redwood` -- measured by one runner, `benchmarks/run.py`. A **profile**
(`benchmarks/profiles.py`) pins every parameter that defines a suite's work.
Only profile runs of a committed tree are kept in the history, since only they
measure the same work on every revision; ad hoc runs are for exploring and
for `just benchmark-compare`.

`just benchmark` runs each routine profile in a fresh process and writes one
report per profile to `runs/<timestamp>/`. Profiles whose data is missing are
skipped (see `just download-data`). Full Redwood is not routine: run it with
`just benchmark --profile redwood --keep`. Options that only select what runs,
such as `--device cuda` or `--repeats 9`, are forwarded; overriding a pinned
value is rejected.

## Baseline environment

Open3D and small-gicp live in the `baseline` dependency group, in a separate
Python 3.12 environment. Set it up once:

```bash
CC=gcc CXX=g++ UV_PROJECT_ENVIRONMENT=.venv-benchmark \
    uv sync --python 3.12 --group baseline
```

`CC`/`CXX` override the compiler that uv's managed CPython reports (clang++),
which the CUDA build rejects. Both environments load the same stable-ABI
extension from the source tree, so rebuilding in either updates both.

`just benchmark-baselines` is needed once per library version or machine, not
per revision. `just benchmark-baseline <suite> --backend <backend>` runs one
suite against one baseline ad hoc. A failing backend does not abort the others,
and operations a backend lacks are skipped with a recorded reason.

## Suites

### Operations

Neighbor search, preprocessing (`voxelize`, `voxelize-reduce`, `normals`,
`statistical-outliers`, `radius-outliers`) and `fpfh`, selected with `--op`:

```bash
just benchmark-run operations --op knn --num-reference 1000000 --k 16
just benchmark-run operations --input redwood --op fpfh --normals data/normals.npy
just benchmark-baseline operations --backend open3d --json runs/operations-open3d.json
```

Search is measured in four scopes: `index-build`, `knn` and `radius` (queries
against a prepared index), and `knn-build-query` (both, from raw points).
Open3D's CUDA kNN builds no index and brute forces each query, so on CUDA only
`knn-build-query` is comparable. torchpcl's radius search is capped at 64
neighbors per query; a radius whose sampled neighborhoods exceed the cap is
rejected.

Parameter mappings to Open3D:

- Outlier filters map `k` directly to `nb_neighbors` / `nb_points`.
- Only `voxelize` plus `reduce(points, "mean")` matches `voxel_down_sample`;
  both are timed.
- Normals compare k-NN to k-NN; FPFH uses bounded hybrid neighborhoods and
  identical input normals.
- small-gicp's voxel downsampling is not benchmarked: multi-threaded, it
  returns a different number of voxels.

The `operations-random` and `operations-redwood` profiles run every operation.
FPFH normals come from `--normals PATH.npy`, the PLY's `nx/ny/nz`, or (for
random input) seeded synthetic unit normals.

### Sampling

`farthest_point_sample` against Open3D's `farthest_point_down_sample` on a
random cloud, all starting from index 0:

```bash
just benchmark-run sampling --num-points 1000000 --num-samples 20000
just benchmark-baseline sampling --backend open3d-tensor
just benchmark-baseline sampling --backend open3d-legacy --device cpu
```

`open3d-tensor` runs on CPU and CUDA at the requested dtype and is the matched
comparison. `open3d-legacy` is CPU-only and always computes in float64.

### Registration

ICP and FGR on pairs with a known rigid transform:

```bash
just benchmark-run registration --data data/ another-pair/
just benchmark-run registration --device cuda --objective point-to-plane
just benchmark-baseline registration --backend open3d
just benchmark-baseline registration --backend small-gicp
```

Each pair directory contains `source.ply`, `target.ply` and
`T_target_source.txt`, a whitespace-delimited 4×4 matrix with
`target = source @ R.T + t`. Clouds may differ in size and overlap partially.

Preprocessing (voxel downsampling, normals and FPFH) is done once by torchpcl,
outside timing, and every backend receives the same arrays; its parameters are
recorded under `preprocessing` in the report. Timing covers the registration
call only. ICP starts from identity; FGR receives no initial pose. small-gicp
has no FGR and estimates its own normals for point-to-plane. Open3D FGR is
CPU/float64 only and its ICP solves the pose in CPU float64; each row records
implementation, device and precisions, so check those before comparing.

Reported per case: median and min/max runtime, peak RAM and VRAM, rotation and
translation error, RMSE against ground truth, and fitness. A solver success
flag means numerical completion, not correct registration. Add
`--open3d-metrics` (baseline environment) for Open3D's fitness and inlier RMSE.

### Redwood

FGR recall on the four Redwood scenes:

```bash
just download-data
just benchmark --profile redwood --keep
just benchmark-run redwood --scenes livingroom1 --device cpu --max-pairs 2   # smoke test
```

Data is read from `data/<scene>-fragments-ply/` and `data/<scene>-evaluation/`.
Pairs with `j <= i + 1` are excluded, and success follows the
[Redwood reference evaluator](https://github.com/qianyizh/ElasticReconstruction/blob/master/Matlab_Toolbox/Core/mrEvaluateRegistration.m).
This measures recall on known overlapping pairs, not loop-closure detection.
`--max-pairs` marks the report as a subset.

## Measurement

Each case gets untimed warmups, then timed repetitions (five by default).
CUDA is synchronized at both timing boundaries. Peak memory comes from one
extra untimed call, sampled every millisecond: RAM is the process's peak RSS,
VRAM the NVML-attributed peak on the selected device. Peaks are approximate
and include the interpreter, libraries and allocator caches. CPU runs default
to the physical core count (`--threads`). CUDA timings vary more than CPU
timings, so compare `time_ms_min`/`time_ms_max` and not just medians.

## Accuracy checking

Accuracy is checked by its own command, never by a benchmark or the test suite:

```bash
just benchmark-accuracy --op all --json runs/accuracy.json
```

It runs no timing and exits nonzero when a check exceeds its tolerance.
Neighbor searches are compared against brute force, everything else against
Open3D on CPU. Rows marked `info` report a value without judging it. Two
tolerances are deliberately nonzero, with reasons in the report: FPFH rows
(float32 histogram bin edges) and outlier masks (threshold summation order).
Outlier checks also require `k±1` to disagree, so agreement at `k` is
meaningful. Open3D is a comparison, not an oracle: re-accept the report after
an Open3D upgrade.

## Comparing reports

```bash
just benchmark-run operations --json runs/before.json
uv sync --group benchmark --reinstall-package torchpcl   # after changing a kernel
just benchmark-run operations --json runs/after.json
just benchmark-compare runs/before.json runs/after.json
just benchmark-compare runs/operations-open3d.json runs/after.json \
    --before-backend open3d --after-backend torchpcl
```

Rows are matched by group, name, backend and device; duplicates are rejected.
Paired backend selectors print speedups (above 1 means the candidate is
faster). A row is marked `*` when it moved more than 2% and the measured ranges
are disjoint -- a heuristic, not a significance test. The comparison does not
check that the inputs match; comparing two runs of one profile ensures it.
`compare.py` needs only the standard library.

## History and dashboard

`just benchmark --keep` (or `just benchmark-history add REPORT...`) stores
profile reports in `benchmarks/history/<machine>/`:

- `torchpcl.jsonl`: one row per case per kept revision.
- `baselines.jsonl`: one row per case per baseline library version.
- `profiles.json`: the pinned values of each profile version.

The machine label is derived from the OS and processor; set
`$TORCHPCL_BENCHMARK_MACHINE` to override it. Use `just benchmark-history`
`list`, `drop REVISION` or `relabel OLD NEW` to manage the history.

Refused: dirty trees, ad hoc runs, accuracy reports, and profiles edited
without bumping their `version` in `profiles.py`. Re-keeping a revision is a
no-op; `just benchmark-history add --replace` overwrites it.

`just benchmark-dashboard` serves an overview of each case's latest time and
factor against the fastest baseline, plus one chart per case across revisions.
It re-renders on reload and binds all interfaces; use `--host 127.0.0.1` to
restrict it, `--port` to move it, or `--out` to write a self-contained HTML
file instead.

## Architecture

- **`Case`** (`case.py`): a backend-neutral dataclass with `run`,
  `synchronize`, `reset`, `evaluate`, optional device memory reader, metadata
  and group.
- **`*_cases.py`**: prepare shared inputs once (`inputs.py`) and build a `Case`
  per backend/device. The only code that calls a specific backend.
- **Suites** (`suites.py`): declare workload arguments and yield cases or skip
  records. `workload.py` is shared with `accuracy.py`.
- **`runner.measure`** (`runner.py`): warmup, repetitions and memory sampling
  (`memory.py`).
- **`run.py`**: the command line and JSON report. `bench.py` runs profiles
  through it; `history.py`, `dashboard.py` and `compare.py` only read reports.

A case can be reused from an external harness:

```python
from benchmarks.registration_cases import RegistrationConfig, prepare_pair, make_case

pair = prepare_pair("data", RegistrationConfig(), with_features=True)
case = make_case(pair, method="fgr", backend="torchpcl", device="cuda")
case.reset()
result = case.run_synchronized()   # time this
scores = case.evaluate(result)
```

### Adding a benchmark

1. Add a `<name>_cases.py` with a `prepare_*` function and
   `make_case(prepared, backend, device, ...) -> Case`; set `evaluate` only if
   there is a ground-truth score.
2. Add a `Suite` to `suites.py` and register it in `SUITES`, unless the
   operation fits an existing suite.
3. Add a `Profile` to `profiles.py` pinning every workload argument.
4. Document it here; add tests only for what `test_benchmark_runner.py` does
   not cover.
