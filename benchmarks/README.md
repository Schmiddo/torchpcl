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
UV_PROJECT_ENVIRONMENT=.venv-benchmark uv sync --python 3.12 --group benchmark
.venv-benchmark/bin/python benchmarks/registration.py --json results-registration.json
```

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

Open3D FGR has no tensor/CUDA implementation in the installed API. It is retained
as a **CPU/float64 reference only**, with `reference_only: true` in JSON; its
runtime is not a matched-device/precision comparison with torchpcl's float32 FGR.
JSON records implementation, device, input and compute precision, and transform
precision. Tensor ICP with float32 points is labeled mixed computation because
of its float64 pose solver. Use `--dtype float64` for double-precision inputs.
Repetitions reset the same
seed to measure runtime variability; use separate `--seed` runs to assess FGR
robustness.

The table reports medians of runtime and accuracy across repetitions, plus peak
memory from one additional untimed registration call:

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

## Sampling benchmark

Compares `torchpcl.farthest_point_sample` with Open3D's
`farthest_point_down_sample` on a random point cloud (no `data/` pair
needed). Uses the same benchmark environment as above:

```bash
.venv-benchmark/bin/python benchmarks/sampling.py --json results-sampling.json
.venv-benchmark/bin/python benchmarks/sampling.py --num-points 1000000 --num-samples 20000
.venv-benchmark/bin/python benchmarks/sampling.py --device cpu --dtype float64
```

Both backends select the same start index (0) on an identical, dtype-quantized
input, so their outputs are directly comparable. Open3D's
`farthest_point_down_sample` is a legacy CPU/float64-only API (`reference_only:
true` in JSON); torchpcl runs on `--device`'s selected device(s) and dtype.
Timing excludes input construction; there is one untimed warmup, five measured
repetitions by default, and one additional untimed call for peak RAM/VRAM
(see "What is measured" above for the memory-sampling methodology, which this
benchmark shares via `benchmarks/memory.py`).

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
  `sampling_cases.py`, and future ones) prepare shared inputs once (loading a
  pair, generating a random cloud, ...) and build a `Case` per
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
