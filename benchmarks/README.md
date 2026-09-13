# Benchmarks

Compare torchpcl with Open3D, operation by operation. Every benchmark is built
from the same `Case`/`measure` contract (see "Adding a new benchmark" below),
so new operations can be added independently of the CLI or of each other, and
the same cases can later back an ASV regression suite without going through
either script's console/JSON output.

## Registration benchmarks

Compare torchpcl with Open3D on pairs with a known rigid transform. Run from
the repository root. Open3D needs Python 3.12 with the current dependency set;
use a separate environment to preserve the project's default Python environment:

```bash
CC=gcc CXX=g++ UV_PROJECT_ENVIRONMENT=.venv-benchmark \
    uv sync --python 3.12 --group benchmark
.venv-benchmark/bin/python benchmarks/registration.py --json results-registration.json
```

`CC`/`CXX` are needed because uv's managed CPython 3.12 is itself built with
clang, so its `sysconfig` names `clang++` as the extension compiler. Without
the override the native build fails ("The current installed version of clang++
(0.0.0) is less than the minimum required version by CUDA 13.0"), even where
clang is absent and PyTorch expects g++ anyway. Re-run this command after
changing anything under `src/torchpcl/csrc/`: an editable install builds
`_C.cpython-312-*.so` in the source tree, and the default environment's
`uv sync --reinstall-package torchpcl` only rebuilds its own interpreter's copy.

The default `data/` pair is local and is not distributed with the repository.
Each pair directory must contain `source.ply`, `target.ply`, and
`T_target_source.txt`, a whitespace-delimited rigid 4×4 matrix mapping source
coordinates into target coordinates: `target = source @ R.T + t`.
Clouds may have different point counts and partial overlap.

```bash
# Additional pairs use the same three filenames.
.venv-benchmark/bin/python benchmarks/registration.py --data data/ another-pair/
.venv-benchmark/bin/python benchmarks/registration.py --method icp --repeats 10
.venv-benchmark/bin/python benchmarks/registration.py --device cpu
.venv-benchmark/bin/python benchmarks/registration.py --device cuda
```

The suite runs two independent cases:

- Point-to-point ICP starts from identity for both libraries, with 50 iterations
  at most, fitness/RMSE change tolerances of `1e-6`, and a correspondence distance
  of five voxel widths. Motion relative to this distance can limit convergence
  from identity, so ground-truth errors should be inspected even on the supplied pair.
- Fast global registration matches shared FPFH descriptors, with 64 iterations,
  tuple scale `0.95`, at most 1,000 tuples, and a correspondence distance of half
  a voxel width. It receives no initial pose or ground-truth correspondences.
  The implementations differ in matching, tuple sampling, and optimization;
  equal seeds do not imply equal random samples across libraries.

Default voxel size is `0.05` in the input's units. Adjust `--voxel-size`,
`--icp-distance`, and `--fgr-distance` for other datasets. See `--help` for options.

### What is measured

Both libraries receive shared Open3D voxel centroids and, for FGR,
Open3D FPFH descriptors. Coordinates and descriptors are rounded to the selected
input dtype (float32 by default) before either backend receives them. Normals use radius `2 * voxel_size` / 30 neighbors and
are oriented toward each cloud's origin; FPFH uses radius `5 * voxel_size` /
100 neighbors. Ground truth is used only for evaluation.

Timing covers the registration call, including internal matching, index building,
and result metrics. Loading, common preprocessing, transfers, and external
accuracy evaluation are excluded. This isolates registration; it does **not**
benchmark torchpcl's voxelization, normals, FPFH, or an end-to-end pipeline.
There is one untimed warmup followed by five measured repetitions. CUDA is
synchronized before and after each timed torchpcl call. By default
(`--device all`), both libraries' ICP cases
run on CPU and CUDA with float32 point inputs. Open3D ICP uses the tensor API;
its point storage and neighbor search use the requested dtype/device, while its
pose solver and output transform use CPU float64 as required by that API.
CUDA is synchronized through the corresponding library on both timing boundaries.
Unavailable CUDA is skipped per library in the console and JSON; an explicit
`--device cuda` request fails if a required CUDA implementation is unavailable.
CPU runs use the physical core count by default; pass `--threads 1` for
single-thread measurements.

Open3D FGR has no tensor/CUDA implementation in the installed API, so it runs
**CPU/float64 only**: its runtime is not a matched-device/precision comparison
with torchpcl's float32 FGR. JSON records implementation (`tensor`, `legacy`,
or `torchpcl`), device, input and compute precision, and transform precision;
read those together to see which rows are directly comparable. Tensor ICP with
float32 points is labeled mixed computation because of its float64 pose solver.
Use `--dtype float64` for double-precision inputs.
Repetitions reset the same
seed to measure runtime variability; use separate `--seed` runs to assess FGR
robustness.

The table reports medians of runtime and accuracy across repetitions, the
fastest repetition, plus peak memory from one additional untimed registration
call. JSON also records `time_ms_min`/`time_ms_max`; compare those before
reading anything into a difference of medians, since CUDA repetitions vary with
clock and thermal state far more than CPU ones do:

- RAM is the peak resident set size of the benchmark process. VRAM is the peak
  allocation attributed by NVML to that process on the selected CUDA device.
  CPU rows have no VRAM value.
- Rotation error in degrees and Euclidean translation error against ground truth.
- GT RMSE: RMS displacement between the full source cloud transformed by the
  estimated pose and by ground truth. This needs no pointwise source/target matches.
- Fitness: fraction of full-resolution source points with a target neighbor
  inside the method's correspondence distance, evaluated identically by Open3D.

Memory is sampled every millisecond, including synchronization, but excluding
loading, shared preprocessing, and case construction. The absolute peaks include
the interpreter, imported libraries, prepared case inputs, and allocator caches;
they represent the process/device capacity required at execution time. JSON also
reports the baseline immediately before registration and the peak increase over
that baseline. Since native allocations may be shorter than the sampling interval,
reported peaks are approximate. Run each case in a fresh process when comparing
absolute memory across machines or integrating memory measurements with ASV.

Translation and RMSE use the PLY's coordinate units. Fitness is threshold-dependent
and should be compared between backends within each method. JSON additionally
records inlier RMSE, every timing/error sample and estimated transform, torchpcl's
FGR solver status, point counts, effective options, and environment versions.
A solver success flag indicates numerical completion, not correct registration.
There are no dataset-independent accuracy pass/fail thresholds: inspect the
ground-truth errors, especially for failed alignments with low inlier RMSE.

## Redwood registration benchmark

Download the four Redwood scenes and compare torchpcl FGR with Open3D FGR:

```bash
just download-data
just benchmark-redwood --json results-redwood.json
# Short CPU smoke test:
just benchmark-redwood --scenes livingroom1 --device cpu --max-pairs 2 --threads 2
```

The runner expects `data/<scene>-fragments-ply/cloud_bin_<index>.ply` and
`data/<scene>-evaluation/{gt.log,gt.info}`, as produced by the download recipe.
Use `--data` to change the root and `--scenes` to select scenes. Each GT record
`i j N` maps source fragment `j` into target fragment `i`; pairs with `j <= i+1`
are excluded. Ground truth is never passed into the registration solver.

This evaluates **registration recall on known overlapping GT pairs**, not
all-pairs loop-closure detection or precision. Correctness uses the
[Redwood reference evaluator](https://github.com/qianyizh/ElasticReconstruction/blob/master/Matlab_Toolbox/Core/mrEvaluateRegistration.m):
form `inverse(T_gt) @ T_est`, concatenate its translation and active quaternion
vector part into `e`, and accept `e.T @ info @ e / info[0,0] <= 0.04`.
This preserves the reference quaternion convention, rather than replacing it
with a rotation vector or raw cloud RMSE. Solver completion alone is not success.

Both cases reuse the registration benchmark's shared voxelization/FPFH and FGR
settings. Timing excludes preprocessing, transfers, and GT evaluation. Open3D
is a CPU/float64 reference; `--device` selects torchpcl devices, defaulting to
CPU and available CUDA. One warmup and one measured repeat are used per pair,
plus the runner's separate memory-sampling call. Set `--repeats`, `--warmup`,
`--dtype`, `--voxel-size`, `--fgr-distance`, `--fgr-iterations`, or `--seed` as needed.
An eight-fragment cache bounds retained preprocessing data.

Console output includes pair timings and success counts. JSON includes per-run
transforms, Redwood errors, pose/fitness metrics, memory, per-scene recall, and
aggregate recall weighted by pair count. Recall is computed separately for each
repeat before averaging. `--max-pairs` selects the first N eligible pairs per
scene and explicitly marks the report as a subset; use it for smoke tests only.

## Sampling benchmark

Compares `torchpcl.farthest_point_sample` with Open3D's
`farthest_point_down_sample` on a random point cloud (no `data/` pair
needed). Uses the same benchmark environment as above:

```bash
.venv-benchmark/bin/python benchmarks/sampling.py --json results-sampling.json
.venv-benchmark/bin/python benchmarks/sampling.py --num-points 1000000 --num-samples 20000
.venv-benchmark/bin/python benchmarks/sampling.py --device cpu --dtype float64
```

All backends select the same start index (0) on an identical, dtype-quantized
input, so their outputs are directly comparable.

Open3D exposes `farthest_point_down_sample` twice, and the two differ by more
than an order of magnitude, so both are benchmarked as separate backends:

- `open3d-tensor` — `t.geometry.PointCloud`, which runs on CPU and CUDA at the
  requested dtype. This is the matched comparison against torchpcl.
- `open3d-legacy` — `geometry.PointCloud`, CPU-only and always computing in
  float64 whatever `--dtype` requests. Despite that, it is currently the faster
  of the two on CPU.

torchpcl and `open3d-tensor` run on `--device`'s selected device(s); Open3D
CUDA availability is resolved separately from torchpcl's and skipped
independently. Each result records `input_dtype` and `compute_dtype`.
Timing excludes input construction; there is one untimed warmup, five measured
repetitions by default, and one additional untimed call for peak RAM/VRAM
(see "What is measured" above for the memory-sampling methodology, which this
benchmark shares via `benchmarks/memory.py`).

## Operation benchmarks

Compare torchpcl's per-operation kernels with Open3D's tensor API on random
clouds (no `data/` pair needed). Neighbor search is covered today; the script
takes an `--op` selector so further operations can be added without another
CLI:

```bash
just benchmark-operations --json results-operations.json
just benchmark-operations --op knn --num-reference 1000000 --k 16
just benchmark-operations --op radius --radius 0.01 --device cuda
```

Four scopes are measured, because the libraries divide the work differently:

- `index-build` — `NeighborIndex(reference)` against
  `NearestNeighborSearch(reference)` plus `knn_index()`. Open3D's constructor
  is lazy, so both calls are inside the timed region.
- `knn` and `radius` — queries only, against an index built during preparation.
- `knn-build-query` — building and querying together, from raw points.

**Read `knn-build-query` before the other three on CUDA.** Open3D's CUDA
`knn_index()` returns in about 0.03 ms for a 200k-point reference, against
55 ms for the same build on CPU: it defers the work to query time rather than
building a structure. Its `index-build` and `knn` figures are therefore not
comparable with torchpcl's in isolation, while the combined figure is what
either library charges to answer a batch of queries from raw points.

Both libraries return **squared** distances. Open3D's radius search is ragged
(flat neighbors plus row splits) where torchpcl returns a fixed-width result
capped at 64 neighbors per query, so a radius whose densest scored
neighborhood exceeds that cap is rejected rather than silently compared
against a truncated result. The console reports the densest neighborhood found.

Preprocessing (`voxelize`, `voxelize-reduce`, `normals`,
`statistical-outliers`, `radius-outliers`) and `fpfh` are measured by the same
script. Their parameter mappings were established by measurement rather than
from documentation:

- Both outlier filters map their neighbor count **directly**: torchpcl's `k`
  is Open3D's `nb_neighbors` / `nb_points`, reproducing its masks exactly,
  despite torchpcl counting the query point among the `k`. `k`±1 disagrees, so
  the mapping is pinned by a test.
- torchpcl's `voxelize` returns a `Voxelization`, not a cloud; only
  `voxelize` plus `reduce(points, "mean")` is like-for-like with Open3D's
  `voxel_down_sample`, which it then reproduces exactly. Both are timed, so
  the reduction's cost is visible instead of quietly omitted.
- Normals are compared k-NN to k-NN. Open3D's optional `radius` changes its
  result, so it is not passed.
- FPFH uses bounded hybrid neighborhoods on both sides. Normals are estimated
  once during preparation and handed to both backends, so only the descriptor
  computation differs.

### Accuracy gates

Every case is scored, and a case scoring zero reproduces the established
implementation exactly.

Search cases are scored against a brute-force reference computed once during
preparation over the first 1024 queries (the full cross product is too large
to materialize at benchmark sizes). `knn` reports the largest absolute
squared-distance error, comparing **sorted** distances because both libraries
leave neighbor order unspecified; `radius` reports the fraction of queries
whose neighbor count disagrees. An index build has nothing to score.

Every other case is scored against Open3D's own CPU result. These report a
**mismatch fraction** rather than a maximum, because a maximum over hundreds
of thousands of rows is dominated by single differing neighborhoods: Open3D's
CPU and CUDA implementations themselves disagree on roughly 0.06% of voxels
and 0.002% of descriptors, which a max renders as a large coordinate error
while a fraction renders it as the small disagreement it is. Normals are
scored on unsigned alignment, since torchpcl's signs are arbitrary without
attached normals.

torchpcl selects search candidates in float32 even for float64 inputs, so
exact agreement is not expected at `--dtype float64`. FPFH additionally
reports `limit_reached_fraction`: a radius holding more than 64 neighbors
means both libraries summarize a truncated neighborhood, and the run says so
rather than letting the comparison look complete.

## Tracking progress across runs

Every script takes `--json PATH`, and nothing is written without it. A report
holds the full environment (including a UTC `timestamp`, the `revision` of the
working tree, and the exact `command`), every tuned parameter under `config`,
and one `results` entry per case with its median, min/max, peak memory,
accuracy gates, and per-repeat samples.

`just benchmark` runs all four comparisons — registration, sampling, and the
operations suite on both the random and Redwood inputs — into
`runs/<UTC timestamp>-<name>.json`. One stamp covers the sweep, so a run's
reports group together and sort chronologically. Extra arguments reach every
script, so only shared ones (`--repeats`, `--device`, `--dtype`) are useful
there; reach for the individual recipes to vary anything else.

```bash
just benchmark                       # full sweep
just benchmark-operations --json runs/before.json   # or one at a time
# ... change a kernel, rebuild both environments ...
just benchmark-operations --json runs/after.json
just benchmark-compare runs/before.json runs/after.json
```

`compare.py` joins rows on name/backend/device and prints the change in median
runtime. A row is marked `*` only when the change exceeds 2% **and** the two
runs' measured `[min, max]` ranges are disjoint; everything else is within the
noise a rerun of an untouched kernel reproduces. Accuracy gates that moved are
printed under their row, so a speedup that broke a result cannot be read as a
win. Parameters that differ between the two reports are listed as `! config`
lines — they mean the comparison is measuring different work, not progress.

A `revision` ending in `-dirty` was measured with uncommitted changes and
cannot be reproduced from the hash. `compare.py` is pure stdlib and runs in
the default environment; only producing a report needs Open3D.

## Architecture

Every benchmark is split into three independent layers so that adding an
operation doesn't touch the others:

- **A `Case`** (`case.py`): a backend/operation-neutral dataclass —
  `run`, `synchronize`, `reset`, `evaluate`, an optional
  `device_memory_reader`, and a `metadata` dict of static descriptive fields
  (dtype, whether a backend is a reference-only baseline, ...). `case.py`
  also holds `select_devices`, the shared `--device all/cpu/cuda` resolution
  used by every benchmark.
- **Per-operation "cases" modules** (`registration_cases.py`,
  `sampling_cases.py`, `neighbors_cases.py`, and future ones) prepare shared
  inputs once (loading a pair, generating a random cloud, ...) -- reusable
  cloud generation itself lives in `inputs.py` -- and build a `Case` per
  backend/device/variant. This is the only place that knows about a specific
  operation's inputs, options, and (if any) accuracy metrics.
- **`runner.measure`** (`runner.py`) times a `Case` with warmup/repetition and
  peak-memory sampling (`memory.py`, backend-neutral). It has no knowledge of
  what a specific operation measures: it medians whatever numeric fields
  `case.evaluate()` returns (an empty dict if there's nothing to score, as for
  sampling) and reports `case.metadata` alongside them.

The CLI scripts (`registration.py`, `sampling.py`) are thin: argument parsing,
picking which cases to build, and console/JSON output. Shared CLI plumbing
(argument types, thread configuration, environment metadata, JSON writing,
`open3d` import guard) lives in `cli.py`; both scripts import it rather than
each defining their own copy. `metrics.py` holds registration-specific
ground-truth validation and pose errors, independent of `runner.py`.

### Reusing the components

An external harness can prepare a case in its setup phase and time only execution:

```python
from benchmarks.registration_cases import RegistrationConfig, prepare_pair, make_case

config = RegistrationConfig()  # voxel_size=0.05, dtype="float32"
pair = prepare_pair("data", config, with_features=True)
case = make_case(pair, method="fgr", backend="torchpcl", device="cuda")

case.reset()                       # reset random seed outside timing
result = case.run_synchronized()    # time this; waits for CUDA completion
scores = case.evaluate(result)      # accuracy evaluation outside timing
```

For ICP-only setup, use `with_features=False`. Prepared pairs can be reused across
methods and devices; each case owns its backend inputs and options. `case.run()`
is the raw registration operation, and `case.synchronize()` exposes the timing
boundary when the harness manages synchronization itself. `case.reset()` resets
Open3D's RNG; torchpcl FGR uses the seed stored in its options on each call.
An external harness owns thread configuration and warmups. This separation lets
an ASV adapter reuse preparation and execution without invoking the CLI or its
wall-clock runner — an ASV `time_*` method just calls `case.reset()` then
`case.run()` (or `case.run_synchronized()`) and lets ASV do its own timing. No
ASV dependency or configuration is required yet.

### Adding a new benchmark

To benchmark another operation (e.g. `voxelize`, `estimate_normals`, `fpfh`):

1. Add a `<name>_cases.py` module with a `prepare_*` function that builds
   shared, backend-independent input once, and a `make_case(prepared,
   backend, device, ...) -> Case` that allocates that backend's inputs/options
   and returns a `Case`. If the operation has a ground-truth score, compute it
   in `evaluate(result)`; otherwise leave `evaluate` unset (defaults to
   returning `{}`, as for sampling).
2. Add a `<name>.py` CLI script mirroring `sampling.py`'s shape (the smaller
   of the two existing scripts): parse operation-specific arguments with
   `cli.py`'s shared helpers, resolve devices with `select_devices`, build one
   `Case` per backend/device via `make_case`, and pass each to
   `runner.measure`.
3. Add a section to this README and, if useful, a `tests/test_benchmark_*.py`
   covering anything not already exercised by `test_benchmark_runner.py`/
   `test_benchmark_memory.py`.
