# torchpcl

Tensor-first point-cloud processing for PyTorch.
Currently supported are neighbor search, voxelization, farthest point sampling,
normals, outlier filtering, FPFH features with fast global registration, and
single- or multi-scale ICP; most ops support packed ragged batches. See the
[API reference](docs/api.md) and [batching guide](docs/batching.md).

## Installation

Binary wheels are available through the
[wheel index](https://schmiddo.github.io/torchpcl/wheels/).
We provide wheels for PyTorch >= 2.10 and Python >= 3.12 using PyTorch's stable
ABI and Python's limited API. Select the index matching your
PyTorch build: `cu130` (CUDA 13), `cu126` (CUDA 12.6+), or `cpu`:

```bash
uv pip install --no-deps torchpcl \
  --find-links https://schmiddo.github.io/torchpcl/wheels/cu130/
```

Earlier releases built per-PyTorch wheels that are available in separate
indexes, such as `wheels/torch2.14.0-cu130/`.

To install from source, clone with the cuBQL submodule:

```bash
git clone --recurse-submodules https://github.com/Schmiddo/torchpcl.git
cd torchpcl
uv sync
```

The extension is compiled against the installed PyTorch, so pip needs build
isolation disabled:

```bash
pip install --no-build-isolation .
```

Build options (environment variables):

- `CUDA_HOME`: CUDA toolkit compatible with the installed PyTorch.
- `TORCHPCL_WITH_CUDA`: `auto` (default; CPU-only with a warning if no toolkit
  is found), `1` to require CUDA, `0` for CPU-only.
- `TORCH_CUDA_ARCH_LIST`: override GPU architecture detection.
- `TORCHPCL_CUBQL_DIR`: use an external cuBQL checkout.

## Point Clouds

An `(N, 3)` tensor represents one cloud, while a `(B, N, 3)` tensor represents
`B` equal-length clouds. Ragged batches use packed points plus offsets:

```python
import torch
import torchpcl as tp

cloud_a = torch.randn(2_000, 3, device="cuda")
cloud_b = torch.randn(3_500, 3, device="cuda")

points = torch.cat([cloud_a, cloud_b])
offsets = torch.tensor([0, len(cloud_a), len(points)], device=points.device)

cloud = tp.PointCloud(points, offsets)
```

Returned neighbor indices refer to rows of the packed point tensor. Padded
data must be converted explicitly:

```python
cloud = tp.PointCloud.from_padded(padded_points, lengths)
padded_points, lengths = cloud.to_padded()
```

## Geometry

Voxelization returns centroids and the reusable point-to-voxel mapping:

```python
partition = tp.voxelize(points, voxel_size=0.05)
downsampled = partition.cloud
voxel_features = partition.reduce(features, reduction="mean")
```

Farthest point sampling instead fixes the output point count, greedily
spreading selections over the cloud (CPU or CUDA, one cloud at a time):

```python
sampled = tp.farthest_point_sample(points, num_samples=1_024)
selected_points = sampled.points       # (1_024, 3), in selection order
```

Normal estimation and covariances work on either downsampled cloud:

```python
normal_result = tp.estimate_normals(downsampled, radius=0.15, k=30)
normals = normal_result.normals        # unit normals; signs match attached
                                       # normals if present, else arbitrary
valid_normals = normal_result.valid    # false where < 3 neighbors were found
curvature = normal_result.curvature    # smallest eigenvalue / eigenvalue sum
```

The same bounded neighborhoods can produce differentiable local moments:

```python
moments = tp.estimate_covariances(downsampled, radius=0.15, k=30)
means = moments.means                  # (P, 3)
covariances = moments.covariances      # (P, 3, 3), sample covariance
valid_moments = moments.valid          # false where < 3 neighbors were found
```

Normal orientation is explicit and can use a shared or per-cloud viewpoint:

```python
cloud = tp.PointCloud.from_points(downsampled.points, normal_result.normals)
oriented_cloud = tp.orient_normals_toward_viewpoint(cloud, viewpoint)
```

`orient_normals_to_direction` uses a fixed direction instead.

Cleaning is mask-first so inspection and selection stay separate:

```python
outliers = tp.radius_outliers(cloud, radius=0.10, k=8)
clean = cloud.select_points(~outliers)
```

The query point counts as a neighbor and the radius boundary is inclusive.
`statistical_outliers` is the per-cloud k-NN distance filter.

## Corresponding-Point Alignment

Use differentiable Procrustes alignment when source and target rows are known
to correspond:

```python
alignment = tp.procrustes(source, target, weights=confidence)
transforms = alignment.transforms       # (B, 4, 4), source-to-target

similarity = tp.procrustes(source, target, estimate_scale=True)
scales = similarity.scale               # (B,)
```

Packed inputs require equal source and target lengths per batch entry. Results
are differentiable; fewer than three positively weighted non-collinear points
are rejected.

## Neighbor Search

Reuse `NeighborIndex` when querying the same reference cloud repeatedly:

```python
index = tp.NeighborIndex(reference, algorithm="auto")
nearest = index.knn(queries, k=8)
local = index.hybrid(queries, radius=0.1, k=32)

nearest.indices       # (Q, K), global packed indices; -1 means invalid
nearest.distances2    # squared distances in the input dtype
nearest.valid         # explicit validity mask
```

Candidates are selected in float32; row order and tie resolution are
unspecified. `tp.knn`, `tp.radius_neighbors`, and `tp.hybrid_neighbors` provide equivalent
one-shot calls.

## Metrics

Metrics operate on tensors or packed clouds and return tensors on the input
device:

```python
per_cloud = tp.chamfer_distance(
    prediction,
    reference,
    squared=False,
    reduction="none",
)

scores = tp.fscore(prediction, reference, threshold=0.01)
metrics = tp.point_cloud_metrics(prediction, reference, threshold=0.05)
```

Results keep the batch dimension, even for an `(N, 3)` input. Chamfer
distances are differentiable, with nearest-neighbor identity treated as
piecewise constant.

## Registration

ICP accepts tensors or packed clouds. A single-scale call contains one
unvoxelized level:

```python
target = tp.PointCloud.from_points(target_points, normals=target_normals)
result = tp.icp(
    source,
    target,
    levels=[
        tp.ICPLevel(
            max_correspondence_distance=0.1,
            max_iterations=30,
        ),
    ],
    options=tp.ICPOptions(objective=tp.PointToPlane()),
)

result.transforms       # (B, 4, 4), source-to-target
result.converged        # (B,), final-level convergence
result.iterations       # (B,), updates accumulated across levels
result.fitness          # (B,), final-level inliers / source length
result.inlier_rmse      # (B,), final-level RMSE
result.level_results    # One diagnostic result per requested level
```

Point-to-plane requires normals attached to the target cloud; ICP never
estimates them implicitly.

Multi-scale ICP uses the same function with several levels:

```python
levels = [
    tp.ICPLevel(
        voxel_size=0.20,
        max_correspondence_distance=0.40,
        max_iterations=30,
    ),
    tp.ICPLevel(
        voxel_size=0.10,
        max_correspondence_distance=0.20,
        max_iterations=20,
    ),
    tp.ICPLevel(
        voxel_size=None,
        max_correspondence_distance=0.10,
        max_iterations=15,
    ),
]
result = tp.icp(source, target, levels)
```

Each level is voxelized from the original clouds; `voxel_size=None` uses them
at full resolution.

`evaluate_registration` evaluates supplied transforms without iteration and
returns `RegistrationMetrics`.

### PLY registration tool

`torchpcl.tools.register` aligns two or more PLY clouds: it runs FPFH-based
global registration for every pair, picks a maximum spanning tree by fitness,
and refines its edges with ICP. The last input defines the reference frame:

```bash
uv run -m torchpcl.tools.register scan-1.ply scan-2.ply scan-3.ply \
    --voxel-size 0.02 \
    --output combined.ply
```

It prints each input-to-reference transform and the ICP metrics. The merged
output keeps PLY colors (uncolored inputs get a random solid color), and a
same-stem `.npz` holds one point-ownership mask per input. All radii default to
multiples of `--voxel-size`; choose it for the units and density of your data.
`--no-global-registration` chains ICP from identity in input order, and
`--icp-objective point-to-plane` uses (or estimates) target normals.

## Behavior

- Geometry supports float32 and float64 on CPU and CUDA; paired inputs must
  share dtype, device, and batch size.
- Neighbor candidates are selected in float32; distances use the input dtype.
- Procrustes, transforms, voxel reductions, and metric distances support
  autograd; normals, FPFH, and ICP are inference-only.
- BVH search and `farthest_point_sample` currently handle one cloud at a time;
  batched search falls back to brute force.

See the [batching guide](docs/batching.md) for empty entries and other details.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

CUDA tests run when a GPU and the CUDA extension are available. Rebuild after
changing native sources:

```bash
uv sync --reinstall-package torchpcl
```

## Benchmarks

Performance can be tracked per revision against Open3D and small-gicp:

```bash
just benchmark --keep           # routine profiles, kept in the history
just benchmark-baselines --keep # Open3D/small-gicp reference values
just benchmark-dashboard        # plot the history
```

See the [benchmark guide](benchmarks/README.md) for suites, measurement scope,
and setup.
