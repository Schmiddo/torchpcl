# Unified ICP API Refactor Plan

## Goal

Make `registration.py` the sole home of the ICP configuration, preprocessing,
solver, and result types. Single-scale and multi-scale registration use one
functional `icp` entry point and differ only in the number and configuration of
their levels.

This is a clean API break. The refactor does not need compatibility aliases,
deprecation paths, or preservation of current behavior and semantics.

Prepared-target and prepared-pyramid registration APIs are out of scope. ICP
accepts source and target clouds directly and constructs the requested levels
internally.

## Target public API

```python
result = tp.icp(
    source,
    target,
    levels=[
        tp.ICPLevel(
            voxel_size=0.20,
            max_correspondence_distance=0.40,
            max_iterations=30,
        ),
        tp.ICPLevel(
            voxel_size=None,  # Use the original clouds.
            max_correspondence_distance=0.10,
            max_iterations=20,
        ),
    ],
    init=initial_transform,
    options=tp.ICPOptions(
        objective=tp.PointToPlane(),
        convergence=tp.ConvergenceCriteria(
            fitness_tolerance=1e-6,
            rmse_tolerance=1e-6,
        ),
        robust_loss=tp.HuberLoss(delta=0.02),
    ),
)
```

Single-scale ICP is the same call with exactly one level:

```python
result = tp.icp(
    source,
    target,
    levels=[
        tp.ICPLevel(
            voxel_size=None,
            max_correspondence_distance=0.10,
            max_iterations=30,
        ),
    ],
)
```

Point-to-plane registration requires normals attached to the target
`PointCloud`. Normal estimation remains explicit through `estimate_normals`;
ICP does not estimate normals automatically.

## API replacement

| Current API | Replacement |
|---|---|
| `ICPScale` | `ICPLevel` |
| `multiscale_icp(...)` | `icp(..., levels=[...])` |
| Single-scale `icp` scalar arguments | One `ICPLevel` |
| `method="point_to_point"` | `PointToPoint()` |
| `method="point_to_plane"` | `PointToPlane()` |
| `robust_kernel`, `robust_delta` | `HuberLoss(delta=...)` |
| `target_normals=` | Normals attached to the target cloud |
| `PointCloudPyramid`, `build_pyramid` | Removed |
| Result classes in `types.py` | Result classes in `registration.py` |
| Public `index=` arguments | Removed |

## 1. Define the public configuration and result model

Add the following public types to `src/torchpcl/registration.py`:

- `ICPLevel`
- `ConvergenceCriteria`
- `ICPOptions`
- `PointToPoint`
- `PointToPlane`
- `HuberLoss`
- `ICPLevelResult`
- `ICPResult`
- `RegistrationMetrics`
- `evaluate_registration`

Use frozen, keyword-only dataclasses for configuration objects.

`ICPLevel` contains:

- `max_correspondence_distance: float`
- `max_iterations: int = 30`
- `voxel_size: float | None = None`

`voxel_size=None` means that the original, unvoxelized source and target are
used for that level.

`ConvergenceCriteria` contains explicitly named absolute tolerances:

- `fitness_tolerance`
- `rmse_tolerance`

`ICPOptions` contains:

- `objective`, defaulting to `PointToPoint()`
- `convergence`, defaulting to `ConvergenceCriteria()`
- `robust_loss`, defaulting to `None`

Configuration objects validate their own scalar fields. The `icp` entry point
validates that its level sequence is nonempty and that it contains only
`ICPLevel` instances. Levels are executed in the order supplied; the API does
not enforce coarse-to-fine voxel ordering.

## 2. Internalize level construction

Normalize and validate the source and target clouds once at the beginning of
`icp`.

For every requested `ICPLevel`:

1. If `voxel_size` is `None`, use the original clouds unchanged.
2. If `voxel_size` is positive, voxelize both original clouds at that
   resolution.
3. For `PointToPlane`, reduce attached target normals into each target voxel
   and renormalize them.
4. Build a private neighbor index for the resulting target level.
5. Package the generated clouds, normals, index, and level configuration in a
   private prepared-level structure.

Each voxelized level is generated independently from the original cloud rather
than progressively downsampling the preceding level.

Missing target normals are an error for `PointToPlane`. Remove implicit normal
estimation and remove the standalone `target_normals` argument. Callers with
tensor inputs can construct a `PointCloud` containing their normals before
calling `icp`.

## 3. Refactor the solver around configuration objects

Replace string-based objective and robust-loss branches with type-based
dispatch:

- `PointToPoint` invokes the packed Procrustes update.
- `PointToPlane` invokes the linearized point-to-plane update.
- `robust_loss=None` uses unweighted valid correspondences.
- `HuberLoss` produces Huber weights using its configured delta.

The private single-level solver accepts a prepared level, the incoming
transforms, and `ICPOptions`. It returns an `ICPLevelResult`.

The sequence runner:

1. Initializes transforms once.
2. Runs each level in the supplied order.
3. Passes the ending transforms from one level to the next.
4. Keeps per-level results.
5. Builds the aggregate `ICPResult`.

Batch entries continue to be solved independently within a level. Convergence
at one level does not prevent an entry from participating in the next level.

## 4. Redesign result aggregation

`ICPLevelResult` contains:

- The corresponding `ICPLevel`
- Ending transforms
- Per-batch convergence flags
- Per-batch iteration counts
- Fitness
- Inlier RMSE

`ICPResult` contains:

- Final transforms
- Final-level convergence flags
- Total per-batch iterations summed across levels
- Final-level fitness
- Final-level inlier RMSE
- `level_results: tuple[ICPLevelResult, ...]`

The final metrics remain convenient for the common case, while
`level_results` makes multi-level behavior observable without retaining
correspondence tensors.

## 5. Replace the functional entry point

Replace the current `icp` signature with:

```python
def icp(
    source: PointCloudLike,
    target: PointCloudLike,
    levels: Sequence[ICPLevel],
    *,
    init: torch.Tensor | None = None,
    options: ICPOptions | None = None,
) -> ICPResult:
    ...
```

The level sequence is required. This keeps correspondence thresholds and
iteration budgets visible at every call site.

Remove these public `icp` parameters:

- `max_distance`
- `method`
- `target_normals`
- `max_iterations`
- `relative_fitness`
- `relative_rmse`
- `robust_kernel`
- `robust_delta`
- `index`

Keep `evaluate_registration` in `registration.py`, rename
`max_distance` to `max_correspondence_distance`, and remove its public `index`
parameter. Its index remains an internal implementation detail.

## 6. Collapse the registration modules

Move the required pyramid preprocessing and all registration result types into
`src/torchpcl/registration.py`.

Organize that file into these sections:

1. Public configuration types
2. Public result types
3. Input normalization and level materialization
4. Correspondence evaluation
5. Objective and robust-loss updates
6. Single-level execution
7. Level-sequence execution
8. Public functions

Then:

- Delete `src/torchpcl/multiscale.py`.
- Delete `src/torchpcl/types.py`.
- Update `src/torchpcl/__init__.py` to import the new public surface from
  `registration.py`.
- Remove `ICPScale`, `PointCloudPyramid`, `build_pyramid`, and
  `multiscale_icp` from `torchpcl.__all__`.
- Do not provide compatibility aliases or deprecated wrappers.

The resulting registration exports are:

- `ConvergenceCriteria`
- `HuberLoss`
- `ICPLevel`
- `ICPLevelResult`
- `ICPOptions`
- `ICPResult`
- `PointToPlane`
- `PointToPoint`
- `RegistrationMetrics`
- `evaluate_registration`
- `icp`

## 7. Replace the registration tests

Merge the useful coverage from `tests/test_multiscale.py` into
`tests/test_registration.py`, then delete `tests/test_multiscale.py`.

Test the new contract directly:

- A single level with `voxel_size=None` uses the original clouds.
- A one-level schedule recovers a small rigid transform.
- A multi-level schedule recovers a transform outside the fine level's basin.
- A schedule can finish with an unvoxelized full-resolution level.
- Point-to-point and point-to-plane objectives both work.
- Point-to-plane requires attached target normals.
- Target normals are reduced and renormalized for voxelized levels.
- `HuberLoss` is accepted and produces finite results.
- Initial transforms are propagated into the first level.
- Ending transforms propagate between successive levels.
- Packed batch entries converge or fail independently.
- Float32 inputs produce float32 transforms and metrics.
- `ICPResult` reports final metrics and total iterations.
- `level_results` contains one correctly configured result per requested level.
- Empty schedules and invalid configuration values are rejected.
- Removed string switches and legacy entry points are absent from the public
  API.

Tests should target the new semantics rather than reproduce the exact
iteration counts or convergence behavior of the old implementation.

## 8. Update documentation and repository call sites

Rewrite the registration sections in:

- `README.md`
- `docs/api.md`
- `REFACTOR_PLAN.md`

Document:

- Single-scale as a one-level schedule
- Multi-scale as a multi-level schedule
- `voxel_size=None`
- Explicit target-normal ownership
- Objective and robust-loss objects
- Final and per-level result fields
- The removal of reusable pyramid inputs

Mechanically update `benchmarks/run_benchmark.py` to invoke the new API so the
repository does not retain a known-broken call site. Do not redesign or run
the benchmarks as part of this refactor, and leave `benchmarks/PLAN.md` and
`benchmarks/BASELINE.md` unchanged for the separate benchmark work.

## Completion criteria

The refactor is complete when:

- `icp` is the only ICP registration entry point.
- Single-scale and multi-scale calls differ only in their `ICPLevel` sequence.
- `multiscale.py` and `types.py` have been deleted.
- No public pyramid or neighbor-index registration inputs remain.
- Point-to-plane normal handling is explicit.
- Every result includes per-level diagnostics.
- The test suite and documentation describe only the new API.
- The full test suite passes.
