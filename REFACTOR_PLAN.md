# torchpcl Refactor Plan

## Objective

Rewrite torchpcl around a small, tensor-first API for batched point-cloud
processing. The target workloads are nearest-neighbor search, voxelization,
normal estimation, point-cloud metrics, and ICP on CPU and CUDA.

This is a breaking refactor. Preserving the current API, numerical details, and
Open3D-compatible behavior is not a goal. It is acceptable for intermediate
commits to support only part of the final matrix, provided the limitation is
documented and each merged operation has focused tests for what it does support.

Priorities, in order:

1. A simple and consistent public API.
2. Correct packed batching without Python loops over clouds.
3. Maintainable implementations and a small native-code surface.
4. Good CPU and CUDA performance without host/device synchronization in hot
   paths.
5. Gradients where they follow naturally from PyTorch operations.
6. Last-mile optimization only when benchmarks justify the complexity.

## Decisions Made Up Front

These decisions should remain fixed through the first complete implementation.
Revisit them only if a phase exposes a concrete problem.

### Breaking release

- Develop the new surface as version `0.2.0`.
- Do not add compatibility wrappers for `PointToPoint`, `PointToPlane`,
  `ICPConvergenceCriteria`, `NearestNeighborSearch`, or the old metric result
  types.
- Delete the old implementation after the replacement passes the relevant
  tests. Do not maintain two complete pipelines.

### Canonical batching representation

- Canonical storage is packed points `(P, 3)` plus integer offsets `(B + 1,)`.
- Offsets are on the same device as the points, start at zero, are
  nondecreasing, and end at `P`.
- Empty batch elements are valid. An entirely empty reference cloud may be
  rejected by operations that cannot define a result for it.
- Public convenience functions also accept an unbatched `(N, 3)` tensor.
- Padded `(B, N, 3)` tensors are converted explicitly with `from_padded`; core
  algorithms do not accept padding and lengths as an alternate internal path.
- Indices into packed data are always global row indices.

### Dtypes and devices

- Public geometry tensors support `float32` and `float64` initially.
- `float16` and `bfloat16` are deferred until accuracy and performance are
  characterized.
- Outputs remain on the input device. Public algorithms do not call `.cpu()`,
  `.item()`, `float(tensor)`, or `int(tensor)` in their hot path.
- Points paired in an operation must have the same device and dtype. Do not
  silently cast one input to another.
- Transforms default to the point dtype. Users may pass higher-precision
  transforms explicitly.

### Gradients

- Transform, voxel feature reductions, and metric distances use normal PyTorch
  autograd where practical.
- Neighbor selection is discrete and non-differentiable. Search indices are
  treated as constant during backward; distances are recomputed with PyTorch
  tensor operations when gradients are requested.
- Normal estimation and ICP are initially inference operations. Do not write
  custom backward kernels for them in the first complete version.
- Do not mark an operation differentiable until `gradcheck` or a focused
  analytical gradient test exists.

### Native code policy

- Prefer vectorized PyTorch implementations for orchestration, reductions, and
  small linear algebra.
- Keep native code for spatial search or an operation proven by benchmarks to
  need it.
- Use PyTorch dispatcher operators (`TORCH_LIBRARY`) for new native entry
  points. Do not add more backend-specific pybind modules.
- A few percent of throughput is not worth duplicated CPU/CUDA algorithms,
  fused kernels with complex backward implementations, or brittle template
  specialization.

## Target Public API

The exact module layout may change during implementation, but this is the
contract to implement and test.

```python
import torchpcl as tp

# Construction
cloud = tp.PointCloud(points, offsets)
cloud = tp.PointCloud.from_points(points)                  # (N, 3)
cloud = tp.PointCloud.from_padded(points, lengths)         # (B, N, 3)
padded, lengths = cloud.to_padded()

# Geometry
moved = tp.transform(cloud, transforms)                    # (B, 4, 4)
voxels = tp.voxelize(cloud, voxel_size=0.05)
normal_result = tp.estimate_normals(cloud, k=30)

# Search
index = tp.NeighborIndex(reference, algorithm="auto")
neighbors = index.knn(queries, k=8)
neighbors = index.radius(queries, radius=0.1, max_neighbors=64)
neighbors = tp.knn(reference, queries, k=8)                # one-shot

# Metrics
distance = tp.chamfer_distance(prediction, reference, reduction="none")
scores = tp.fscore(prediction, reference, threshold=0.05)

# Registration
result = tp.icp(
    source,
    target,
    method="point_to_plane",
    target_normals=normals,
    max_distance=0.1,
)
```

Core result types:

```python
@dataclass(frozen=True)
class PointCloud:
    points: Tensor              # (P, 3)
    offsets: Tensor             # (B + 1,)
    normals: Tensor | None = None
    features: Tensor | None = None

@dataclass(frozen=True)
class Neighbors:
    indices: Tensor             # (Q, K), global indices, -1 if invalid
    distances2: Tensor          # (Q, K)
    valid: Tensor               # (Q, K), bool

@dataclass(frozen=True)
class Voxelization:
    cloud: PointCloud
    coordinates: Tensor         # (V, 4): batch, x, y, z
    point_to_voxel: Tensor      # (P,)
    counts: Tensor              # (V,)

@dataclass(frozen=True)
class NormalResult:
    normals: Tensor             # (P, 3)
    valid: Tensor               # (P,), bool
    curvature: Tensor | None

@dataclass(frozen=True)
class ICPResult:
    transforms: Tensor          # (B, 4, 4)
    converged: Tensor           # (B,), bool
    iterations: Tensor          # (B,), integer
    fitness: Tensor             # (B,)
    inlier_rmse: Tensor         # (B,)
```

Result dataclasses contain tensors and metadata only. They must not trigger a
device synchronization in constructors or properties.

## Phase 0: Establish the Rewrite Scaffold

Status: implemented on 2026-07-12. CPU baseline measurements are recorded in
`benchmarks/BASELINE.md`; CUDA measurements remain pending because no usable GPU
was available on the baseline host.

### Tasks

1. Add this plan to version control and create a short architecture section in
   the README linking to it.
2. Record current benchmark results for search, voxel downsampling, normals,
   Chamfer, and ICP. These are reference numbers, not hard regression limits.
3. Reorganize source files around the target concepts:
   - `cloud.py`: packed data model and conversions;
   - `neighbors.py`: search interface and result type;
   - `voxel.py`: voxelization and reductions;
   - `normals.py`: normal estimation;
   - `metrics.py`: tensor-valued metrics;
   - `registration.py`: ICP;
   - `transforms.py`: batched rigid transforms;
   - `validation.py`: shared validation helpers.
4. Add a test helper for generating ragged packed batches, including empty
   batch elements.
5. Add a benchmark helper that synchronizes only around timed regions and
   reports CPU and CUDA separately.

### Completion gate

- Existing code still imports, or the branch clearly documents that the rewrite
  is temporarily incomplete.
- The new module layout and test utilities exist without duplicating algorithm
  implementations.
- Baseline benchmark output is saved in a short Markdown table with hardware and
  software versions.

## Phase 1: Implement `PointCloud` and Batched Transforms

This phase fixes the representation used by every later operation.

Status: implemented on 2026-07-12.

### Tasks

1. Implement `PointCloud` validation:
   - points are floating `(P, 3)` tensors;
   - offsets are integer `(B + 1,)` tensors on the same device;
   - normals and features, if present, have leading dimension `P`;
   - structural offset validation avoids CUDA synchronization where possible.
2. Implement `from_points`, `from_padded`, `to_padded`, `batch_size`,
   `lengths`, `device`, `dtype`, `to`, and `clone`.
3. Decide and document whether metadata tensors are copied or shared. Prefer
   immutable dataclasses with tensor storage shared unless an explicit copy is
   requested.
4. Implement `transform` for:
   - an unbatched tensor and one `(4, 4)` transform;
   - a `PointCloud` and `(B, 4, 4)` transforms;
   - normals, using rotation only when normals are attached.
5. Implement an internal `batch_ids(offsets)` helper. Cache it only after
   profiling shows repeated construction matters.
6. Add gradient tests for points and transforms.

### Completion gate

- Packed/padded round trips preserve valid points and lengths.
- Empty batch entries round trip correctly.
- Transform behavior is tested on CPU and CUDA when available.
- `torch.autograd.gradcheck` passes for a small float64 transform case.

## Phase 2: Unify Neighbor Search

Build one batched interface over the existing BVH and brute-force code before
changing search kernels.

Status: implemented on 2026-07-12. Native dispatcher consolidation remains in
Phase 8; the current BVH backend remains restricted to a single reference
cloud, while packed batches use the brute-force backend.

### Tasks

1. Replace the separate `NearestNeighborSearch` and
   `BatchedNearestNeighborSearch` public classes with `NeighborIndex`.
2. Make `NeighborIndex` accept `PointCloud` or `(N, 3)` and retain the packed
   points and offsets.
3. Implement `knn`, `radius`, and `hybrid` with the same packed query contract.
4. Standardize results:
   - global `int64` indices;
   - squared distances in the input dtype;
   - explicit boolean validity mask;
   - ascending distance order;
   - no semantic dependence on the distance stored in invalid slots.
5. Keep `algorithm="bruteforce"` and `algorithm="bvh"` explicit. Make
   `"auto"` initially choose BVH for reusable indices and brute force only for
   small one-shot cases after measuring a crossover.
6. Add one-shot `knn`, `radius_neighbors`, and `hybrid_neighbors` functions.
7. Wrap the current extensions first. Only then register a single dispatcher
   operator such as:

   ```text
   torchpcl::knn(points, point_offsets, queries, query_offsets, k, radius)
       -> (indices, distances2)
   ```

8. Move CPU and CUDA implementations behind this dispatcher operator. Retire
   the brute-force pybind modules once both backends work.
9. Keep cuBQL behind an internal implementation boundary. Do not expose cuBQL
   types or constraints in the public API.
10. Recompute differentiable distances from gathered points in a small Python
    helper. Add gradient tests for those distances, not for neighbor selection.

### Completion gate

- One API handles single clouds and ragged batches on CPU and CUDA.
- Tests cover batch isolation, global indices, empty queries, finite radius,
  missing neighbors, and tied distances.
- Search does not loop over batch entries in Python.
- Importing torchpcl never triggers compilation.
- Existing search classes and their tests are removed after all new tests pass.

## Phase 3: Replace Voxel Downsampling with Voxelization

Implement this primarily with PyTorch sorting and scatter reductions. Add a
native kernel only if the benchmark gap is large enough to matter.

Status: implemented on 2026-07-12 with PyTorch unique and scatter reductions.

### Tasks

1. Compute signed integer voxel coordinates with `floor(points / voxel_size)`.
   Do not use the current fixed 21-bit packed coordinate key.
2. Include batch ID in the key and lexicographically sort `(batch, x, y, z)`.
   Prefer stable, understandable tensor operations over a custom hash table.
3. Produce unique voxel coordinates, inverse `point_to_voxel`, and counts.
4. Compute centroid points with `index_add_` or `scatter_reduce`.
5. Add `Voxelization.reduce(values, reduction=...)` with initial reductions
   `mean`, `sum`, `min`, `max`, and `first`. Implement only `mean` and `sum`
   in the first commit if that keeps the change focused.
6. Preserve attached features and normals only when an explicit reduction is
   selected. Do not guess how categorical or directional data should combine.
7. Define deterministic output ordering as lexicographic voxel-coordinate
   order.
8. Add a `voxel_downsample` convenience alias only if it remains useful; it
   should simply return `voxelize(...).cloud`.
9. Test negative and very large coordinates, ragged batches, empty entries,
   feature gradients, and CPU/CUDA agreement.

### Completion gate

- Voxelization cannot mix points across batches.
- There is no coordinate-range limit caused by bit packing.
- Mean and sum reductions have tested gradients.
- Performance is measured against the old function. Accept a modest slowdown
  if it removes native code or coordinate restrictions.

## Phase 4: Rebuild Normal Estimation on Unified Search

Status: implemented on 2026-07-12 using `torch.linalg.eigh`. Normal estimation
is intentionally inference-only.

### Tasks

1. Implement `estimate_normals` using `NeighborIndex.hybrid` or `.knn`.
2. Use one straightforward covariance implementation shared by CPU and CUDA.
3. Start with `torch.linalg.eigh` for the batched `3x3` solve. Keep the current
   analytic eigensolver only if benchmarks demonstrate that it materially
   improves realistic workloads and its numerical tests are strong.
4. Make self-neighbor inclusion explicit and default to inclusion.
5. Return `NormalResult` with a validity mask rather than silently relying on a
   zero vector.
6. Support orientation modes:
   - `"none"`;
   - `"viewpoint"` with viewpoint `(3,)` or `(B, 3)`.
7. Return curvature optionally from the covariance eigenvalues.
8. Initially run under `torch.no_grad()` or document the operation as
   non-differentiable. Reconsider gradients only if a real use case appears.

### Completion gate

- Plane, sphere, isolated-point, radius-limited, and ragged-batch tests pass.
- CPU and CUDA results agree up to normal sign when orientation is disabled.
- No Python loop exists over clouds or points.
- Normal estimation reuses a caller-provided `NeighborIndex` when supplied.

## Phase 5: Make Metrics Batched and Tensor-Valued

Status: implemented on 2026-07-12 with packed segmented reductions and
piecewise-differentiable gathered distances.

### Tasks

1. Implement a shared `directed_nearest_distance` primitive using unified
   search and differentiable gathered-distance recomputation.
2. Implement `chamfer_distance` with explicit options:
   - `squared: bool`;
   - `directional: "both" | "source_to_target" | "target_to_source"`;
   - `point_reduction: "mean" | "sum"`;
   - `reduction: "none" | "mean" | "sum"` over batches.
3. Implement segmented reductions over packed points. Avoid converting offsets
   to Python lists or looping over batches.
4. Implement `fscore` with scalar or tensor thresholds. Return precision,
   recall, and F-score tensors per batch and threshold.
5. Add Hausdorff distance and normal consistency only after Chamfer and F-score
   are stable.
6. Replace Python-float metric dataclasses with tensor-valued result types.
7. Add gradient tests for squared and unsquared Chamfer, including exact point
   matches and ragged batches.
8. Document that gradients do not pass through changes in nearest-neighbor
   identity.

### Completion gate

- Metrics support unbatched and ragged batched clouds through one code path.
- `reduction="none"` returns `(B,)` or `(B, T)` tensors on the input device.
- Chamfer gradients are finite at zero distance and tested on CPU and CUDA.
- No per-cloud Python loop or implicit host synchronization remains.

## Phase 6: Implement Batched ICP

Favor a readable PyTorch loop over iterations. Fuse kernels only if later
profiling identifies a compelling bottleneck.

Status: implemented on 2026-07-12. Correspondences remain internal and are not
part of `ICPResult`. The loop reads one aggregate active flag per iteration to
avoid running searches after all batch entries have converged or failed.

### Tasks

1. Define a functional API with simple values rather than estimator classes:
   - `method="point_to_point" | "point_to_plane"`;
   - `max_distance`, `max_iterations`, `relative_fitness`, `relative_rmse`;
   - optional per-batch initial transforms;
   - optional target normals.
2. Build or accept a reusable target `NeighborIndex` once per call.
3. Track active batch elements with a boolean tensor. Converged or failed batch
   entries stop updating while other entries continue.
4. Compute correspondences, counts, squared-error sums, and normal equations
   with segmented tensor reductions.
5. Implement batched point-to-point Kabsch with `torch.linalg.svd` on `(B, 3,
   3)` matrices.
6. Implement batched point-to-plane using `(B, 6, 6)` systems and
   `torch.linalg.solve_ex`. Define a stable fallback for singular systems:
   mark the batch entry failed and retain its current transform.
7. Keep all convergence statistics as device tensors. A fixed-length Python
   iteration loop is acceptable; values must not be read back to Python each
   iteration.
8. Define result semantics:
   - source-to-target transforms;
   - fitness is valid correspondences divided by source length;
   - RMSE is zero for no correspondences;
   - failed entries retain the last valid transform;
   - `converged=False` on failure or iteration exhaustion.
9. Add robust weighting only after basic ICP works. Start with Huber; defer
   trimming, reciprocal correspondences, colored ICP, and generalized ICP.
10. Treat ICP as non-differentiable initially. Do not add custom autograd.
11. Implement `evaluate_registration` using the same correspondence and metric
    helpers, not a separate pipeline.

### Completion gate

- Single-cloud and ragged-batch point-to-point ICP pass synthetic recovery
  tests.
- Point-to-plane works with packed target normals.
- Batch entries can converge or fail independently.
- The iteration loop has no `.cpu()` or `.item()` calls.
- CPU and CUDA produce comparable transforms on representative cases.

## Phase 7: Add Multi-Scale ICP

### Tasks

1. Add an `ICPScale` dataclass containing voxel size, maximum correspondence
   distance, and iteration count.
2. Implement `multiscale_icp` as composition of `voxelize`,
   `estimate_normals`, and `icp`.
3. Carry each scale's output transform into the next scale.
4. Allow precomputed pyramids and normals so repeated registration does not
   redo preprocessing.
5. Keep scale orchestration in Python. Do not create a monolithic native
   multi-scale operator.

### Completion gate

- A synthetic case outside single-scale ICP's convergence basin is recovered by
  the multi-scale pipeline.
- Batched inputs work through the same API.
- Precomputed and internally computed pyramids give equivalent results.

## Phase 8: Consolidate Packaging and Native Dispatch

Do this after the desired API is exercised end to end, so packaging work does
not optimize interfaces that are still changing.

### Tasks

1. Register remaining native search kernels with `TORCH_LIBRARY` CPU and CUDA
   implementations.
2. Add Meta/FakeTensor implementations for custom operators where feasible.
3. Add autograd registration only for native outputs that promise gradients.
4. Collapse extension build configuration into the smallest practical number
   of modules. CPU-only installation must remain supported.
5. Make optional CUDA availability discoverable through a small capability API,
   for example `tp.backends.cuda.is_available()`.
6. Remove dead pybind bindings, legacy classes, obsolete tests, and unused
   third-party code only after verifying it has no remaining caller.
7. Test wheel installation in a clean CPU environment and a CUDA environment.

### Completion gate

- A clean install imports without runtime compilation.
- CPU-only installation has no CUDA toolkit requirement.
- The public API does not expose backend-specific extension modules.
- A minimal `torch.compile` smoke test works for transforms, voxel feature
  reduction, and Chamfer. ICP compilation is optional.

## Phase 9: Documentation, Examples, and Release Gate

### Tasks

1. Rewrite the README around four complete examples:
   - unbatched voxelization and normals;
   - ragged batched Chamfer;
   - batched ICP;
   - reusable neighbor index.
2. Document packed offsets and global-index semantics in one dedicated page or
   README section.
3. Add API documentation for empty inputs, invalid neighbors, dtypes, devices,
   gradients, determinism, and result direction conventions.
4. Add a migration note that states the old API was intentionally removed. A
   symbol-by-symbol compatibility guide is unnecessary.
5. Run the full correctness suite and benchmark suite on CPU and CUDA.
6. Profile representative workloads before adding any further optimization.
7. Tag `0.2.0` only when all operations in the minimal surface work for packed
   batches on CPU and CUDA, except explicitly documented optional backends.

### Completion gate

- A new user can perform all four target workflows from README examples.
- Tests exercise every public function with an unbatched and packed-batch case.
- Known unsupported dtype/device/gradient combinations fail with clear errors.
- Benchmark results and accepted regressions are documented.

## Cross-Cutting Test Strategy

Every operation should use the same compact test matrix rather than accumulating
large backend-specific suites:

1. A hand-checkable unbatched case.
2. A ragged batch with at least two different lengths and one empty entry when
   the operation permits it.
3. CPU/CUDA comparison when CUDA is available.
4. `float32` and one focused `float64` test.
5. Invalid shape/device/dtype tests at the public boundary.
6. Gradient tests only for operations that promise gradients.

Use pure PyTorch or small brute-force reference implementations in tests. Do
not use Open3D behavior as the specification. Open3D comparisons may remain as
informational integration tests.

## Benchmark Policy

Benchmark before optimizing and retain simple implementations unless a result is
materially poor for an intended workload.

Measure:

- CPU and CUDA separately;
- single clouds and ragged batches;
- end-to-end time, including index construction for one-shot operations;
- index query time separately when reuse is expected;
- peak CUDA memory where practical;
- synchronization outside the timed region only.

Initial representative sizes:

- small: 512 to 2,048 points per cloud;
- medium: 10,000 to 50,000 points per cloud;
- large: 100,000 or more points per cloud;
- batch sizes: 1, 8, and 32 with deliberately uneven lengths.

Optimization is justified when it produces a meaningful end-to-end improvement
on one of these cases without substantially increasing maintenance cost. Avoid
optimizing microbenchmarks that do not affect voxelization, normals, metrics, or
ICP.

## Intentionally Deferred

The following are outside the first complete rewrite:

- approximate neighbor search;
- differentiating through neighbor identity, normals, or ICP;
- half-precision geometry;
- distributed point-cloud batches;
- streaming or out-of-core clouds;
- generalized, colored, non-rigid, or probabilistic ICP;
- custom CUDA kernels for voxelization and segmented reductions;
- aggressive fusion of the ICP correspondence and solve pipeline;
- guaranteed bitwise determinism across CPU and CUDA;
- preserving Open3D-specific edge-case behavior.

## Recommended Delivery Order

Implement and merge in this order:

1. Scaffold and baseline measurements.
2. `PointCloud` plus batched transforms.
3. Unified neighbor search.
4. Voxelization.
5. Normal estimation.
6. Tensor-valued metrics.
7. Batched point-to-point ICP.
8. Point-to-plane and robust ICP.
9. Multi-scale ICP.
10. Dispatcher and packaging consolidation.
11. Documentation and `0.2.0` release cleanup.

The first useful vertical slice is steps 1 through 6. At that point the data
model and shared geometry operations are usable even before ICP is rewritten.
Do not start native performance rewrites or multi-scale ICP before that slice is
stable.
