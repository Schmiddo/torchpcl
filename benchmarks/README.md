# Benchmarks

Track torchpcl performance per revision, against Open3D and small-gicp as
reference baselines. Run commands from the repository root.

```bash
uv sync --group benchmark
just benchmark                  # every routine profile on this revision
just benchmark --keep           # ... and keep it in the history (committed tree only)
just benchmark-baselines --keep # Open3D/small-gicp reference values, per library version
just benchmark-dashboard        # plot the history on :8770
just benchmark-run operations --op knn --k 16   # ad hoc: any suite, any parameters
```

There are four suites -- `operations`, `sampling`, `registration` and
`redwood` -- and one runner, `benchmarks/run.py`, that measures any of them.
A **profile** (`benchmarks/profiles.py`) pins every parameter that defines a
suite's work: sizes, radii, dtype, seed, `threads=8`, warmup. Only profile
runs of a committed tree are kept, because only they measure the same work on
every revision; an ad hoc run with other parameters is for exploring and
comparing with `just benchmark-compare`.

`just benchmark` runs the routine profiles in fresh processes and writes one
report per profile into `runs/<timestamp>/`. Profiles whose data is missing
are skipped with a note (`just download-data`), and full Redwood (739 pairs) is
not routine: run it with `just benchmark --profile redwood --keep`. Options
that only select what runs -- `--device cuda`, `--repeats 9`, an operation
subset -- reach every run; anything a profile pins is rejected.

Baselines do not change with torchpcl's commits, so `just benchmark-baselines`
is needed once per library version (or machine, or profile change), not per
revision. It runs in the Python 3.12 baseline environment, measures every
third-party backend of every routine profile, and each backend picks the
devices it supports. A backend that fails does not abort the others. Its ad
hoc sibling `just benchmark-baseline <suite> --backend <backend>` runs one
suite against one backend.

The `benchmark` dependency group contains memory tooling, without Open3D or
small-gicp, which live in `baseline`. Each operation/sampling invocation selects
one `--backend`, defaulting to `torchpcl`. small-gicp implements neighbor search
and normals; a sweep skips the operations it lacks with a recorded reason, while
an explicit `--op` for one of them is an error. Its voxel downsampling is
deliberately not benchmarked: with more than one thread it collects points in
parallel and returns a different number of voxels -- 4935 against 4751 at 200k
points, up to 10% by its own documentation -- so it is not equivalent work. Only the selected backend's device availability is checked.
`--device all` records unavailable CUDA as skipped; explicitly requesting an
unavailable device fails. Performance runs report time and memory, and compute
no reference output of any kind. The test suite needs neither Open3D nor a
baseline environment; registration evaluation still has its own scoring.

Every suite shares the `Case`/`measure` contract described below and runs
in the same environment. Accuracy is checked by its own command, never by
a benchmark and never by the test suite; see "Accuracy checking" below.

## Registration benchmarks

Measure ICP and FGR on pairs with a known rigid transform:

```bash
just benchmark-run registration --json runs/registration.json
```

Preprocessing -- voxel downsampling, normals and FPFH -- is torchpcl's and runs
in the project environment, so this needs no Open3D. What it produces is the
solver's input, not a measurement: it sits outside every timed region and both
backends receive the same arrays unchanged. The parameters are recorded under
`preprocessing` in each pair's report, including `feature_neighbors`, which is
64 because that is torchpcl's maximum. That is a recorded choice rather than a
reproduction of any other library's; changing it changes the workload and
invalidates earlier baselines.

`--objective` selects the ICP residual: `point-to-point` (the default) or
`point-to-plane`, which uses the shared normals. small-gicp is the exception --
its `PointCloud` has no normal setter, so it estimates its own from the same
points, and a point-to-plane row against it compares two normal implementations
rather than one shared array. The comparison tool says so when the rows differ.

Open3D and small-gicp remain available for their baseline roles, both in the
Python 3.12 baseline environment:

```bash
just benchmark-baseline registration --backend open3d      # the baseline to beat
just benchmark-baseline registration --backend small-gicp  # a strong CPU baseline
just benchmark-baseline registration --open3d-metrics     # adds fitness/inlier RMSE
```

All three converge to the same pose: within a hundredth of a degree for
point-to-point, and torchpcl and Open3D agree to four decimals on point-to-plane
because they share the same normals. That agreement is what makes the timing
comparison meaningful, and unlike the timings it is reproducible anywhere.

One quiet-machine session on the local pair, 8 threads and 28k points per cloud,
put small-gicp ahead on both objectives -- roughly 2x on point-to-point and
close to 9x on point-to-plane. Re-measure before relying on the figures: a
contended run of the same commands moved every absolute number by more than an
order of magnitude and reordered torchpcl against Open3D, while leaving
small-gicp's lead intact. Read `time_ms_min` next to the median.

small-gicp is CPU only and has no global registration, so it runs ICP alone;
`--method all` skips FGR with a recorded reason. It is multi-threaded by
design, so `--threads` is part of its workload rather than an environment
detail and is passed to every one of its calls.

`--open3d-metrics` needs Open3D, so run it through
`just benchmark-baseline registration --open3d-metrics`. Set up the environment
once with:

```bash
CC=gcc CXX=g++ UV_PROJECT_ENVIRONMENT=.venv-benchmark \
    uv sync --python 3.12 --group baseline
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
just benchmark-run registration --data data/ another-pair/
just benchmark-run registration --method icp --repeats 10
just benchmark-run registration --device cuda --objective point-to-plane
```

The `registration` profile is point-to-point ICP and FGR on the `data/` pair.

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

Download the four Redwood scenes and measure FGR recall against their ground
truth:

```bash
just download-data
just benchmark --profile redwood --keep   # all four scenes, kept
# Short CPU smoke test:
just benchmark-run redwood --scenes livingroom1 --device cpu --max-pairs 2 --threads 2
```

Ground truth comes from the dataset, not from another library, so recall needs
no baseline implementation and this runs in the project environment. Use
`--backend open3d` through `just benchmark-baseline redwood` for the comparison
baseline.

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

Both cases reuse the registration benchmark's shared preprocessing and FGR
settings. Timing excludes preprocessing, transfers, and GT evaluation. Open3D
FGR, when selected, is a CPU/float64 baseline; `--device` selects devices,
defaulting to CPU and available CUDA. One warmup and one measured repeat are used per pair,
plus the runner's separate memory-sampling call. Set `--repeats`, `--warmup`,
`--dtype`, `--voxel-size`, `--fgr-distance`, `--fgr-iterations`, or `--seed` as needed.
An eight-fragment cache bounds retained preprocessing data.

Console output includes pair timings and Redwood errors. JSON includes per-run
transforms, Redwood errors, pose/fitness metrics, memory, per-scene recall
under `input.scenes`, and aggregate recall weighted by pair count under
`summary`; the history keeps only that summary. Recall is computed separately for each
repeat before averaging. `--max-pairs` selects the first N eligible pairs per
scene and explicitly marks the report as a subset; use it for smoke tests only.

## Sampling benchmark

Measures `torchpcl.farthest_point_sample` or an explicitly selected Open3D
`farthest_point_down_sample` backend on a random cloud (no dataset needed):

```bash
just benchmark-run sampling --json runs/sampling.json
just benchmark-run sampling --num-points 1000000 --num-samples 20000
just benchmark-run sampling --device cpu --dtype float64

# Optional baselines, after setting up the baseline environment above:
just benchmark-baseline sampling --backend open3d-tensor
just benchmark-baseline sampling --backend open3d-legacy --device cpu
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
CUDA availability is checked only when that backend is selected. The legacy
backend rejects an explicit CUDA request and records CUDA as unsupported for
`--device all`. Each result records `input_dtype` and `compute_dtype`.
Timing excludes input construction; there is one untimed warmup, five measured
repetitions by default, and one additional untimed call for peak RAM/VRAM
(see "What is measured" above for the memory-sampling methodology, which this
benchmark shares via `benchmarks/memory.py`).

## Operation benchmarks

Measure neighbor search, preprocessing or FPFH with `--op`, defaulting to all
operations. Use the same arguments when recording an Open3D baseline:

```bash
just benchmark-run operations --json runs/operations.json
just benchmark-run operations --op knn --num-reference 1000000 --k 16
just benchmark-run operations --op radius --radius 0.01 --device cuda
just benchmark-baseline operations --backend open3d --json runs/operations-open3d.json
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
capped at 64 neighbors per query. A radius whose densest sampled neighborhood
exceeds that cap is rejected. The console reports the sampled maximum; this
check over at most 1,024 queries does not prove all neighborhoods are untruncated.
Capped and unbounded timings remain contextual comparisons.

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
- FPFH uses bounded hybrid neighborhoods on both sides. Identical input normals
  are handed to both backends, so only descriptor computation is timed.

### Points and FPFH normals

The `operations-random` and `operations-redwood` profiles run every operation
on each input. Random inputs are seeded and quantized to the requested dtype. `--input redwood`
reads points and optional normals together with `torchpcl.load_ply` (backed by
`plyfile`), without Open3D. Use `--fragment PATH` to select a
cloud after downloading data with `just download-data`.

FPFH needs shared normals, supplied in this order:

1. `--normals PATH.npy`: an `(N, 3)` array in the same order as the input points.
2. For PLY input, the file's `nx/ny/nz` properties.
3. For random input, synthetic unit normals from a separate seeded stream.

Normals must be finite and nonzero and match the points' shape. A PLY without
valid normals needs an explicit normals array for FPFH; other operations do not.
Loading and dtype conversion are outside timing. The report records the normals
source and seed/path. Synthetic normals are performance inputs, not estimated
surface normals. **Refresh old FPFH baselines:** the earlier benchmark estimated
normals with Open3D, so its workload differs. To preserve an old workload,
supply its normals explicitly to both backend runs.

```bash
just benchmark-run operations --input redwood --op knn
just benchmark-run operations --input redwood --op fpfh --normals data/normals.npy
```

Performance runs do not compute Open3D reference outputs or call scoring hooks.
Neighbor preparation brute forces nothing on its own: a timing run measures a
search rather than checking one. The scored prefix is computed only when
something asks for it -- the radius capacity guard, which needs neighbor counts
and so runs for `radius` alone, and the accuracy tool below. At the default
200k reference that moved 3.5s out of every kNN and index-build run.

## Accuracy checking

A benchmark answers "how fast". Whether torchpcl still agrees with an accepted
implementation is a separate question with its own command, because combining
them makes each worse: a scoring pass runs on measured output, correctness
failures surface as odd timings, and every performance run drags a reference
computation behind it.

```bash
just benchmark-accuracy --op all --json runs/accuracy.json
```

It runs no timing, prints one table and exits nonzero when a check exceeds its
tolerance. Two references are used, because they are not equally strong:
neighbor searches are compared against **brute force**, the true answer, and
everything else against **Open3D on CPU**, the accepted implementation. Rows
marked `info` report a number without judging it — the fraction a filter
removed, the worst FPFH bin, whether a bounded search truncated — because a
comparison on a workload where the operation did nothing is trivially
satisfied and should be visible as such.

Tolerances are the accepted part, and each carries its reason in the report.
Two are deliberately nonzero, for reasons measured rather than assumed:

- **FPFH** allows 1e-4 of rows. A feature landing on a histogram bin edge
  rounds either way in float32 and moves one neighbor's contribution, about
  `100/k` of the 600 per row, between adjacent bins. At 200k random points two
  rows differed, each a clean transfer between bins 29 and 30 with mass
  conserved, none near the radius boundary and none truncated.
- **Outlier masks** allow 1e-4 of points. Both filters threshold a statistic
  accumulated over the whole cloud, and summation order differs between
  implementations, so the thresholds differ by more than a few ULP. On a 196k
  Redwood fragment two points differed, each ~300 float32 ULP from the
  threshold. A wrong `k` mapping moves 1e-3 to 6e-2 of all points instead,
  which the `offset_k_mismatch_fraction` check pins explicitly: `k±1` must
  disagree, or agreement at `k` would evidence nothing.

The accuracy run defaults `--outlier-radius` to 0.03 on random input rather
than the benchmark's 0.05. At 200k uniform points 0.05 removes six of them, so
both libraries agree on "keep everything" and the comparison proves nothing;
0.03 removes 25%. Real fragments discriminate at their own default. Timing
defaults are unchanged, since altering them would invalidate saved baselines.

The discriminating check needs a workload dense enough to have something to
disagree about, which the default sizes provide; a much smaller `--num-reference`
can leave the filter removing nearly everything or nearly nothing, and the check
then reports honestly that the run cannot evidence the mapping.

Open3D is a comparison implementation, not a universal oracle. A check above
tolerance means the two disagree and someone must look, not that torchpcl is
wrong. Re-accept the report after an Open3D upgrade rather than assuming
continuity.

## Tracking progress across runs

Every run accepts `--json PATH`, and `just benchmark` writes one per profile.
A report holds the environment (timestamp, revision, command, versions,
machine/platform/GPU), `suite`, `profile` (name, version and pinned values, or
`null` for an ad hoc run), `config`, `input` (what was prepared), `skipped`,
and one `results` row per case with median, min/max, memory and per-repeat
samples. Registration and Redwood rows also carry their quality metrics and a
`group` naming the pair they ran on.

### Comparing two reports

```bash
just benchmark-run operations --json runs/before.json
# Change a kernel and rebuild the project environment's native extension:
uv sync --group benchmark --reinstall-package torchpcl
just benchmark-run operations --json runs/after.json
just benchmark-compare runs/before.json runs/after.json

# Reuse a separately recorded Open3D baseline:
just benchmark-compare runs/operations-open3d.json runs/after.json \
    --before-backend open3d --after-backend torchpcl
```

Without selectors, `compare.py` matches group/name/backend/device and shows
the percentage change in median runtime. `--backend NAME` restricts both
reports. With paired `--before-backend` / `--after-backend` selectors, it
matches the selected rows by group/operation/device and prints
baseline/candidate speedup and both measured ranges. A speedup above 1 means
the candidate was faster. Rows present only in one report remain visible.

The comparison prints available environment provenance and `! config`
differences; it does not validate compatibility. Choose matching inputs,
parameters, threads and hardware yourself -- or compare two runs of one
profile, which fixes all of them. Point/normal files must also have unchanged
content, not just matching paths. The printed notes call out unequal work
such as bare voxelization, CUDA index-build splits, capped radius queries and
legacy precision. Such ratios are contextual timings, not equivalent-work
speedups.

A row gets `*` when the change exceeds 2% and both measured ranges are
disjoint (`compare.moved`, the same rule the dashboard uses). This is a
heuristic, not statistical significance; an unmarked row does not prove
absence of regression. Changed accuracy metrics are listed as `gate:` lines.

Each row must be uniquely identified by group, name, backend and device; a
report containing the same row twice is rejected rather than silently compared
against whichever copy landed last. `compare.py` uses only the standard
library, so it runs without project dependencies or either backend:

```bash
python3 benchmarks/compare.py runs/before.json runs/after.json
```

### Keeping a history and a dashboard

`just benchmark --keep` (or `just benchmark-history add REPORT...`) promotes
profile reports into `benchmarks/history/<machine>/`, which outlives `runs/`:

- `torchpcl.jsonl` holds one row per case per kept revision: the series.
- `baselines.jsonl` holds one row per case per baseline library version.
  Re-measuring the same version replaces its row; a new version adds one.
- `profiles.json` records the pinned values each profile version was first
  kept with.

Whether that directory is committed, kept locally or synced some other way is
an open question; the format is plain JSONL either way. The machine is a label
for the computer that measured the points, derived from its operating system
and processor (`linux-amd-ryzen-9-7950x`), because timings compare only within
one computer. Two boxes with the same processor would collide; set
`$TORCHPCL_BENCHMARK_MACHINE` on one of them, or rename a stored label with
`just benchmark-history relabel OLD NEW`. `just benchmark-history list` shows
what is stored; `just benchmark-history drop REVISION` removes a point you
regret.

Refused rather than stored:

- **A dirty revision.** A point measured on an uncommitted tree cannot be
  attributed to the commit it names. `just benchmark --keep` checks this
  before measuring anything.
- **An ad hoc run.** Without a profile there is no guarantee the next revision
  measures the same work, so there is no series to join.
- **A profile edited without a version bump.** Series are keyed on profile
  name and version. Changing a pinned value is a new workload: bump `version`
  in `profiles.py`, which starts new cards and hides the old ones as
  superseded. Pinning a newly added argument at the value it always had is the
  same work and needs no bump.
- **An accuracy report**, which measures agreement, not speed.

Re-keeping the same revision is a no-op; `--replace` overwrites its points.

`dashboard.py` renders the history into one page. An overview matrix at the
top lists each case's newest torchpcl time for every profile and device on
one line (a suite with several profiles, like `operations`, gets its own
block; the rest share one), with
its factor against the fastest baseline (coloured only when the measured
ranges are disjoint) and any movement since the previous revision; each cell
links to its card. Below it the cards sit in one collapsible section per
device, CUDA first and collapsed on load: one card per case of a
profile version, torchpcl's series across revisions with its measured min/max
as a band, and each baseline as a dashed reference line at its newest
measurement, labelled with its library version. Superseded profile versions
are hidden unless the checkbox asks for them. The banner at the top lists only
movement `compare.moved` accepts. Redwood contributes its per-backend summary
(mean per-pair time and recall) rather than hundreds of per-pair rows.

By default it serves that page over HTTP and re-renders it on every request,
so a newly kept point needs a browser reload rather than a restart, and the
history can be read from a laptop while the benchmarks run on the machine with
the GPU. It binds every interface for that reason; `--host 127.0.0.1` keeps it
on the local machine, and `--port` moves it off 8770 when something already
has that one.

```bash
just benchmark-dashboard                      # http://<this machine>:8770/
just benchmark-dashboard --host 127.0.0.1     # this machine only
just benchmark-dashboard --out                # write runs/dashboard.html, don't serve
```

With `--out` it writes the same page to a file instead. That copy embeds its
data and loads nothing, so it needs no server and no network and can be
attached to a message.

### When to refresh a baseline

Refresh a performance baseline when the workload, the adapter, the hardware or
a relevant runtime setting changes. Age alone does not invalidate one, and
recent controlled runs from separate environments stay comparable as long as
their provenance and workload differences are explicit. Accuracy reports have a
different trigger: re-accept them after an Open3D upgrade. Neither kind
invalidates the other.

## Architecture

Every benchmark is split into independent layers so that adding an operation
doesn't touch the others:

- **A `Case`** (`case.py`): a backend/operation-neutral dataclass --
  `run`, `synchronize`, `reset`, `evaluate`, an optional
  `device_memory_reader`, a `metadata` dict of static descriptive fields
  (dtype, implementation, ...) and an optional `group` naming the input it ran
  on. `case.py` also holds `select_devices`, the shared `--device all/cpu/cuda`
  resolution.
- **Per-operation "cases" modules** (`registration_cases.py`,
  `sampling_cases.py`, `neighbors_cases.py`, ...) prepare shared inputs once
  -- reusable cloud generation lives in `inputs.py` -- and build a `Case` per
  backend/device/variant. This is the only place that knows how to call a
  specific backend. Agreement between backends is not their concern:
  `accuracy.py` owns every comparison, and builds its cases through the same
  `make_case` the benchmark uses, so a check cannot drift from what is
  measured.
- **Suites** (`suites.py`) declare a suite's workload arguments, prepare its
  inputs and yield its cases, or skip records naming what a backend cannot
  run. `workload.py` holds the operation arguments and input preparation that
  the operations suite and `accuracy.py` share.
- **`runner.measure`** (`runner.py`) times a `Case` with warmup/repetition and
  peak-memory sampling (`memory.py`). It medians whatever numeric fields
  `case.evaluate()` returns, and only suites scored against ground truth
  (registration, Redwood) ask it to evaluate.
- **`run.py`** is the one command line: backend loading, device selection,
  profile pinning, the console table and the JSON report, for every suite.
  `bench.py` runs profiles through it, one process each; `history.py`,
  `dashboard.py` and `compare.py` only read reports and need no torch.

`cli.py` holds shared plumbing (argument types, thread configuration,
environment metadata, JSON writing, import guards); `metrics.py` holds
registration's ground-truth validation and pose errors.

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

To benchmark another operation:

1. Add a `<name>_cases.py` module with a `prepare_*` function that builds
   shared, backend-independent input once, and a `make_case(prepared,
   backend, device, ...) -> Case` that allocates that backend's inputs/options
   and returns a `Case`. If the operation has a ground-truth score, compute it
   in `evaluate(result)`; otherwise leave `evaluate` unset.
2. Add a `Suite` subclass to `suites.py`: its workload arguments, `backends`,
   any `selection` arguments a profile should leave open, and a `cases()`
   generator yielding one `Case` per backend/device. Register it in `SUITES`.
   An operation that fits an existing suite's inputs belongs in that suite
   instead, as the operations suite's neighbor, preprocessing and FPFH cases
   do.
3. Add a `Profile` to `profiles.py` pinning every workload argument; parsing
   the profile fails until it does.
4. Add a section to this README and, if useful, a `tests/test_benchmark_*.py`
   covering anything not already exercised by `test_benchmark_runner.py`.
