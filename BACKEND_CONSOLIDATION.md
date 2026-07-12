# Backend Consolidation Proposal

## Recommendation

Consolidate the four current pybind extensions into one private extension module,
`torchpcl._C`, while keeping the public API and most policy in Python.

The native backend should provide only two mechanisms:

1. a reusable single-cloud BVH index;
2. stateless packed brute-force k-NN.

Do not make PyTorch dispatcher integration a goal of this work. In particular,
do not add `TORCH_LIBRARY`, custom autograd, Meta/FakeTensor kernels,
`torch.compile` accommodations, export support, or public native operators.
Those features can be reconsidered later if a concrete use case justifies their
maintenance cost.

## Why Consolidate

The current backend is split across:

- `_cubql_cpu`;
- `_cubql_cuda`;
- `_bruteforce_cpu`;
- `_bruteforce_cuda`.

This duplicates bindings, validation, import fallbacks, device selection, and
build definitions. It also leaks the extension layout into `search.py` and
causes `NeighborIndex` to cache complete wrapper objects by radius even though a
BVH itself is radius-independent.

The CUDA packed kernel additionally copies offsets to the host on every query
to validate them and calculate its launch shape. That is both a synchronization
point and a consequence of the current native interface rather than an
algorithmic requirement. The Python layer adds one more per-call
synchronization on top: `NeighborIndex` reads a reduction over batch lengths
back to the host to raise on empty reference batches that have queries.

The consolidation should improve structure first. It should not become a broad
rewrite of every search algorithm or every Torch integration boundary.

## Design Principles

### Python owns policy

Python should continue to own:

- `PointCloud` and packed-batch validation;
- `algorithm="auto"` selection;
- float64-to-float32 search conversion;
- global int64 result indices;
- exact distance recomputation in the input dtype;
- radius enforcement against recomputed distances;
- autograd through gathered distances;
- all-invalid results for empty references and empty reference batches;
- `Neighbors` construction and public error messages.

### Native code owns mechanisms

Native code should own:

- building and retaining a cuBQL BVH;
- CPU and CUDA candidate search;
- tiled packed brute-force k-NN;
- current CUDA stream use and device guards;
- low-level tensor shape, dtype, contiguity, and device assertions.

### The native API is private

`torchpcl._C` is not a supported user API. Its inputs may assume they were
constructed by the validated Python layer. This permits a smaller native
contract without trying to make arbitrary direct `_C` calls safe or ergonomic.

### One implementation per responsibility

CPU and CUDA necessarily have different kernels, but they should share:

- one binding module;
- one C++ interface;
- one result contract;
- one set of constants and validation messages where practical.

Avoid template frameworks or generalized spatial-index abstractions unless a
second real implementation needs them.

## Proposed Native API

The complete pybind surface should be approximately:

```python
import torchpcl._C as _C

index = _C.BvhIndex(points_f32)
indices_i32, approximate_distances2_f32 = index.knn(
    queries_f32,
    k,
    radius,
)

indices_i32, approximate_distances2_f32 = _C.packed_knn(
    points_f32,
    point_offsets_i64,
    queries_f32,
    query_batch_ids_i64,
    k,
    radius,
)

_C.has_cuda() -> bool
```

No other native object needs to be public, even inside the private module.

### `BvhIndex`

- Accepts one non-empty contiguous float32 `(N, 3)` tensor.
- Selects its CPU or CUDA implementation from the tensor device.
- Builds the BVH once; radius is supplied to each query.
- Supports `1 <= k <= 64`.
- Dispatches `k == 1` internally to cuBQL's `findClosest` instead of the
  k-NN candidate walk. 1-NN dominates ICP and Chamfer, and the current
  `findClosest` fast path is only reachable through the legacy wrappers being
  deleted, so consolidation must move it inside `knn` rather than lose it.
- Returns int32 indices and float32 candidate distances.
- Rejects packed batches. Packed BVH support is a separate future feature.

### `packed_knn`

- Accepts contiguous float32 points and queries.
- Accepts point offsets and one batch ID per query.
- Searches only the point range belonging to each query batch.
- Returns global int32 point indices.
- Pads missing neighbors with `-1` on CPU and CUDA alike.
- Returns all-invalid rows for queries whose reference batch is empty; an
  empty point range is a normal input, not an error.

Passing query batch IDs instead of query offsets permits a flat query launch.
The CUDA kernel no longer needs to copy offsets to the host to calculate a
two-dimensional grid or maximum per-batch query count. CPU work can similarly
parallelize over all queries without nested batch loops.

`PointCloud` already derives batch IDs without a host read when the total output
size is supplied to `torch.repeat_interleave`.

## Result Contract

The native result is intentionally narrower than public `Neighbors`:

- indices are int32;
- distances are float32 search distances;
- invalid indices are `-1`;
- each row contains the k nearest candidates within the radius, in no
  guaranteed order;
- invalid distance slots have no public semantic guarantee.

The Python layer converts indices to int64, gathers reference points, recomputes
exact distances, applies the radius in the input dtype, and builds the validity
mask. This keeps gradients and user-visible semantics out of native code.

### No ordering guarantee

Ordering is deliberately not part of the contract, native or public. Which
candidates are returned — the k nearest within the radius — is guaranteed;
their order within a row and the resolution of distance ties are not. The
kernels happen to produce ascending distances today, but CPU, CUDA, BVH, and
brute force are free to differ, and no internal consumer selects by position
(normals reduce over all neighbors; ICP, Chamfer, and the metrics use
`k == 1`). Consumers select by the validity mask. Tests that compare
implementations against each other or against `torch.cdist` must compare
order-insensitively.

### Empty reference batches

Querying against an empty reference batch stops being an error. The flat
kernels naturally return `-1` for every slot of such queries (their point
range is empty), and the Python layer marks them invalid like any other
unmatched query. This replaces the current `ValueError` — and with it the
per-call device-to-host synchronization in `NeighborIndex._validate_queries`.
For a completely empty single-cloud reference, the adapter returns all-invalid
results without constructing a `BvhIndex`; the point count is known from the
tensor shape, so this also needs no synchronization.

Downstream operations already consume the validity mask, so they see an empty
batch as a batch without correspondences; verify during implementation that
normals, metrics, and ICP rely only on the mask. Update the empty-batch
policy tests to expect all-invalid results instead of an exception.

## C++ Structure

Use a small source tree with explicit ownership:

```text
src/torchpcl/csrc/
  bindings.cpp
  backend.h
  common/
    candidate.h
    checks.h
  cpu/
    bvh.cpp
    packed_knn.cpp
  cuda/
    bvh.cu
    packed_knn.cu
```

### `backend.h`

Declare only:

- the device-independent `BvhIndex` wrapper;
- CPU factory/functions;
- CUDA factory/functions behind `TORCHPCL_WITH_CUDA`;
- `packed_knn` device dispatch.

Use a small private implementation interface or PImpl for BVH state. The
binding source should not include cuBQL implementation headers. With CPU and
CUDA sources linked into one module, `CUBQL_GPU_BUILDER_IMPLEMENTATION` must
be defined in exactly one translation unit: `cuda/bvh.cu`.

### `bindings.cpp`

Bind `BvhIndex`, `packed_knn`, and `has_cuda`. Device dispatch belongs in the
C++ wrapper, so Python never imports device-specific modules.

### `common/candidate.h`

Define the fixed-capacity top-k selection helper shared by the CPU and CUDA
brute-force kernels. It guarantees which candidates are kept — the k nearest —
not how they are ordered; the BVH path uses cuBQL's own candidate handling and
need not match. Keep it header-only and independent of pybind and cuBQL.

Do not generalize this into a reusable container library. It only needs to
support the fixed maximum `k` used by search.

### Validation

The native boundary should check facts available without reading device data:

- dtype;
- rank and shape;
- contiguity;
- common device;
- `k` and radius;
- point count fitting the chosen index type.

Offset monotonicity and query batch ID correctness are validated when the
public `PointCloud` is constructed or when Python derives them. The private
native call should not copy CUDA metadata to the host on every query. Drop the
current "a batch with queries must contain at least one point" checks; empty
point ranges are handled by the result contract, not by an error.

If stronger defense is later needed, add an optional debug validation kernel.
Do not put synchronous validation in the release hot path merely to support
unsupported direct calls to `_C`.

## Python Structure

Replace `search.py` with a private adapter such as `_backend.py`:

```text
neighbors.py       public search behavior and result semantics
_backend.py        tensor normalization and calls into _C
_C                 private compiled module
```

`NeighborIndex` should hold one of:

- `_C.BvhIndex` for one-cloud BVH search;
- packed float32 points and offsets for stateless brute-force search.

Remove the radius-keyed backend cache. A BVH is constructed once and queried at
different radii; brute force has no index to cache.

After index construction, the query path must perform no device-to-host reads:
the empty-batch check leaves with its error policy, batch IDs are derived with
`torch.repeat_interleave(..., output_size=...)`, and remaining Python-side
checks use only shapes and scalar arguments.

The adapter should provide one clear error when a CUDA tensor is used with a
CPU-only build. `has_cuda()` is sufficient for this internal capability check;
a larger public backend namespace is unnecessary.

## Build Configuration

Build exactly one module named `torchpcl._C`.

### CPU-only build

Use `CppExtension` with:

- `bindings.cpp`;
- CPU BVH and packed k-NN sources;
- cuBQL include path;
- optimization and supported parallel-runtime flags.

### CUDA build

Use `CUDAExtension` with the same binding and CPU sources plus both CUDA
sources. Define `TORCHPCL_WITH_CUDA=1` for the whole target.

The Python module name and API must be identical in both builds.

### Build selection

Replace the ambiguous current behavior with one setting:

```text
TORCHPCL_WITH_CUDA=auto  # default
TORCHPCL_WITH_CUDA=0     # force CPU-only
TORCHPCL_WITH_CUDA=1     # require CUDA or fail clearly
```

In `auto`, use `torch.utils.cpp_extension.CUDA_HOME`; build CPU-only when no
toolkit is available. Continue to allow `TORCHPCL_CUBQL_DIR`.

An `auto` build must state unmistakably in its output which variant it
produced, and should warn when the installed torch is itself a CUDA build but
no toolkit was found — a silently CPU-only wheel on a GPU machine is a support
ticket waiting to happen (build nodes without `nvcc` are common on clusters).
`TORCHPCL_WITH_CUDA=1` remains the way to turn that situation into a hard
error.

Prefer PyTorch's standard `TORCH_CUDA_ARCH_LIST` over another project-specific
architecture variable.

Remove the `py_limited_api="cp39"` wheel option: it is a mistagging bug, not
an option pending verification. The extensions use pybind11 via
`torch/extension.h`, which does not build under the stable ABI, so the abi3
tag lets pip install the wheel on Python versions it was never built for and
fail at import.

Platform-specific OpenMP flags should be isolated in one build helper. Carry
over the existing `setup.py` note that `-fopenmp` is required for
`at::parallel_for` to parallelize at all: ATen's OpenMP backend is
header-inline, so without the flag queries silently run single-threaded. If
portable parallel configuration becomes complicated, document Linux as the
initial supported native-build platform rather than accumulating fragile
compiler detection.

## CUDA Kernel Cleanup

The packed CUDA launch should be one-dimensional over queries:

1. Each thread owns one query.
2. `query_batch_ids[query]` identifies its point range.
3. Point offsets remain on-device.
4. Reference points are scanned in shared-memory tiles.
5. Candidate selection uses the shared comparison helper.
6. Launch on PyTorch's current stream under `CUDAGuard`.

This removes:

- the host offset copy;
- the 65,535 batch-grid limit;
- host calculation of maximum queries per batch;
- separate per-batch launch geometry.

Do not add warp-specialized variants, multiple `k` template families, or
approximate search during consolidation. Profile the clean kernel first.

## CPU Kernel Cleanup

Parallelize directly over query ranges. Each query reads its batch ID and point
offset range, then runs the same fixed-capacity top-k logic as CUDA.

Retain straightforward scalar C++ unless profiling shows a material regression.
Explicit SIMD, architecture intrinsics, and multiple tile specializations are
not consolidation prerequisites.

## Delivery Sequence

### 1. Freeze the private contract

- Add tests for the proposed `_C` signatures and top-k selection
  (order-insensitive).
- Port the still-relevant legacy search test cases (meta-device rejection,
  batch isolation, radius boundaries) instead of rewriting them from scratch.
- Keep all public `NeighborIndex` tests as the behavioral specification,
  updated for the two sanctioned semantic changes: no ordering guarantee and
  the empty-batch result policy.
- Document native float32/int32 limitations as internal details.

### 2. Build the unified CPU module

- Add the new source layout and `_C` CPU target.
- Implement CPU BVH and packed k-NN behind the new binding.
- Switch CPU `NeighborIndex` to `_backend.py`.
- Compare correctness and benchmark results with the legacy modules.

### 3. Add CUDA to the same module

- Add CUDA BVH factory and flat packed k-NN kernel.
- Build the same `_C` module with `CUDAExtension`.
- Switch CUDA `NeighborIndex` to `_backend.py`.
- Run the full CUDA suite and stream/device tests.

### 4. Remove the legacy layout

- Delete the four old extension definitions and binding entry points.
- Delete `search.py` and obsolete backend-specific tests.
- Keep focused native mechanism tests plus public search tests.
- Confirm source and wheel installs contain only `_C`.

### 5. Packaging verification

- Test a clean forced CPU build without a CUDA toolkit.
- Test a clean required-CUDA build.
- Test `auto` in both environments.
- Confirm import performs no compilation and produces no extension-cache files.

### 6. Benchmark and tune conservatively

- Compare BVH construction, repeated queries, packed brute force, normals,
  Chamfer, ICP, and multi-scale ICP.
- Benchmark `k = 1` explicitly: it is the dominant query in ICP and Chamfer
  and must not regress against the current `findClosest` path.
- Include a many-batch packed case, both to validate the flat-launch gains
  and to keep the deferred packed-BVH gap visible.
- Fix material regressions caused by the consolidation.
- Keep optimizations only when they have clear benchmark value and do not
  obscure ownership boundaries.

## Tests

### Native mechanism tests

Keep these small:

1. BVH k-NN against `torch.cdist` on CPU and CUDA (order-insensitive).
2. Packed k-NN batch isolation and global indices.
3. Radius boundary and invalid padding.
4. Empty queries and empty reference batches (all-invalid rows).
5. CPU/CUDA agreement on the selected candidate set, compared as sets.
6. Current CUDA stream and non-default device smoke tests.

### Public behavior tests

Continue to rely on `tests/test_neighbors.py` and downstream operation tests for:

- float32 and float64 public distances;
- gradients through gathered distances;
- ragged batching;
- the empty-batch policy (all-invalid results, no error);
- normals, metrics, ICP, and multi-scale integration.

Do not duplicate the complete public matrix at the native boundary.

### Installation tests

At minimum, automate:

- CPU-only editable install;
- CPU-only wheel install in a clean environment;
- CUDA editable or wheel install on a CUDA runner;
- import and one search query after installation.

## Acceptance Criteria

- Only `torchpcl._C` is compiled and imported.
- CPU-only and CUDA builds expose the same private API.
- No runtime compilation occurs.
- Public search and downstream tests pass, with exactly two sanctioned
  semantic changes: results carry no ordering guarantee, and empty reference
  batches yield invalid neighbors instead of an error.
- No device-to-host transfer occurs anywhere on the CUDA query path — native
  or Python — after index construction.
- `k = 1` queries are at least as fast as the current `findClosest` path.
- BVH construction is independent of query radius.
- Public CPU and CUDA distances remain in the input dtype and differentiable.
- Build configuration has one clear CUDA selection policy.
- Legacy extension modules and wrapper classes are removed.
- Benchmark regressions are measured and explained before merge.

## Explicit Non-goals

- `TORCH_LIBRARY` or dispatcher registration;
- Meta or FakeTensor kernels;
- `torch.compile`, export, or graph-capture guarantees;
- custom native autograd;
- public access to native operators;
- native float64 search;
- ordering or tie-breaking guarantees, within or across implementations;
- padded native inputs;
- packed BVH construction;
- approximate nearest neighbors;
- new ICP or voxelization kernels;
- aggressive CUDA specialization or kernel fusion.

The result should be a backend that is easy to locate, build, test, and reason
about. Broader Torch integration should only be added later in response to a
specific requirement, not as a prerequisite for consolidating search.
