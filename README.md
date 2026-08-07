# torchpcl

Tensor-first point-cloud processing for PyTorch.
Currently supports packed ragged batching and fast nearest-neighbor search on CPU and CUDA.

The public API covers neighbor search, voxelization, normal estimation,
point-cloud metrics, and single- or multi-scale ICP. See the [API reference](docs/api.md),
[batching guide](docs/batching.md), and [migration guide](MIGRATION.md).

## Installation

Extensions are compiled at installation time against the PyTorch in the active
environment. Build isolation must be disabled:

```bash
pip install --no-build-isolation .
```

A CUDA build requires `CUDA_HOME` to point to a toolkit compatible with the
installed PyTorch. The default `TORCHPCL_WITH_CUDA=auto` builds CUDA when such a
toolkit is found and otherwise builds CPU-only with an explicit warning. To
force a CPU-only build:

```bash
TORCHPCL_WITH_CUDA=0 pip install --no-build-isolation .
```

Use `TORCHPCL_WITH_CUDA=1` to require CUDA and fail installation when the
toolkit is unavailable.

Set `TORCH_CUDA_ARCH_LIST` to override local GPU architecture detection or
`TORCHPCL_CUBQL_DIR` to use an external cuBQL checkout.

Published source distributions include the required cuBQL headers. When
installing from a Git checkout, initialize the submodule first:

```bash
git submodule update --init third_party/cuBQL
```

NumPy, plyfile, and Trimesh are only needed by the repository's file-processing
tools and optimization example. Install them with:

```bash
pip install --no-build-isolation ".[tools]"
```

## Point Clouds

An `(N, 3)` tensor represents one cloud, while a `(B, N, 3)` tensor represents
`B` equal-length clouds. Ragged batches use packed points plus offsets:

```python
import torch
import torchpcl as tp

cloud_a = torch.randn(2_000, 3, device="cuda")
cloud_b = torch.randn(3_500, 3, device="cuda")
points = torch.cat([cloud_a, cloud_b])
cloud = tp.PointCloud(
    points,
    torch.tensor([0, len(cloud_a), len(points)], device=points.device),
)
```

Tensor batches are flattened to packed row order without copying when their
storage permits it. Offsets delimit contiguous clouds and returned neighbor
indices refer to global rows in the packed point tensor. Variable-length
padded data must be converted explicitly:

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

Cleaning is mask-first so inspection and selection stay separate:

```python
outliers = tp.radius_outliers(
    cloud,
    radius=0.10,
    min_neighbors=8,
    include_self=False,
)
clean = cloud.select_points(~outliers)
```

`statistical_outliers` provides the corresponding per-cloud k-NN distance filter.

## Corresponding-Point Alignment

Use differentiable Procrustes alignment when source and target rows are known
to correspond:

```python
alignment = tp.procrustes(source, target, weights=confidence)
transforms = alignment.transforms       # (B, 4, 4), source-to-target

similarity = tp.procrustes(source, target, estimate_scale=True)
scales = similarity.scale               # (B,)
```

Tensor inputs represent one cloud or an equal-length batch. Packed `PointCloud`
inputs support ragged batches and require equal source and target lengths in
every batch entry.
Rotation, translation, scale, and weights participate in ordinary PyTorch
autograd. Degenerate correspondence sets with fewer than three positively
weighted non-collinear points are rejected.

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

The selected candidates are the nearest under the backend's float32 search
coordinates, but their row order and tie resolution are unspecified. Squared
distances are recomputed from the original tensors in their input dtype.
Queries paired with an empty reference batch receive all-invalid rows.

`tp.knn`, `tp.radius_neighbors`, and `tp.hybrid_neighbors` provide equivalent
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

F-score and combined metric fields always retain the batch dimension, including
the implicit batch of size one created by an `(N, 3)` tensor. F-score accepts
one scalar threshold per call.

Chamfer distances are differentiable with nearest-neighbor identity treated as
piecewise constant. Convert padded batches to `PointCloud` before computing
metrics.

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

Point-to-plane requires normals attached to the target cloud. Estimate them
explicitly with `estimate_normals` when they are not already available.
Correspondences and neighbor indices remain internal.

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

Every voxelized level is constructed from the original clouds.
`voxel_size=None` uses the original clouds directly, which also allows a
multi-scale schedule to finish at full resolution.

`evaluate_registration` evaluates supplied transforms without iteration and
returns `RegistrationMetrics`.

## Behavior

- Geometry supports float32 and float64 on CPU and CUDA.
- Neighbor candidate selection uses float32 coordinates for both input dtypes.
  Returned squared distances use the input dtype, so float64 inputs retain
  float64 distance evaluation but not float64 candidate selection.
- Paired inputs must have identical dtype, device, and batch size.
- Procrustes alignment, transforms, voxel reductions, and gathered metric
  distances support autograd. Search indices are discrete; normals and ICP are
  inference-only.
- CPU and CUDA may choose different indices when float32 search distances tie.
- Empty batch entries are supported by storage, transforms, voxelization, and
  search queries. Metrics and registration reject empty cloud pairs.
- Multi-cloud batches currently use brute-force search; BVH indexing currently
  supports a single reference cloud.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

CUDA tests are collected when `torch.cuda.is_available()` and the CUDA search
extensions are installed. After changing native sources, rebuild with:

```bash
uv sync --reinstall-package torchpcl
```

## Benchmarks

```bash
uv run python benchmarks/run_benchmark.py --task all --device cpu --repeats 5
uv run python benchmarks/run_benchmark.py --task registration --repeats 5
uv run python benchmarks/run_benchmark.py --task search --search-sizes 512 2048 8192
uv run python benchmarks/run_benchmark.py --task all --json results.json
```

The registration task contains alignment-only one-level cases and end-to-end
multi-level cases. The latter include voxelization, explicit point-to-plane
normal estimation, neighbor indices, and every ICP level. Configure the
schedule with `--level-voxels`, `--level-distances`, and
`--level-iterations`; all three lists must have the same length.

Install the `benchmark` dependency group and pass `--comparisons` to add
small_gicp and Open3D performance rows. Recorded baseline results are in
[benchmarks/BASELINE.md](benchmarks/BASELINE.md).
