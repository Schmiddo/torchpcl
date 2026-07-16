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

## Point Clouds

An `(N, 3)` tensor is accepted for ordinary single-cloud operations. Ragged
batches use packed points plus offsets:

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

Offsets delimit contiguous clouds and returned neighbor indices refer to global
rows in the packed point tensor. Padded data must be converted explicitly:

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

## Corresponding-Point Alignment

Use differentiable Procrustes alignment when source and target rows are known
to correspond:

```python
alignment = tp.procrustes(source, target, weights=confidence)
transforms = alignment.transforms       # (B, 4, 4), source-to-target

similarity = tp.procrustes(source, target, estimate_scale=True)
scales = similarity.scale               # (B,)
```

Tensor inputs represent one cloud. Packed `PointCloud` inputs support ragged
batches and require equal source and target lengths in every batch entry.
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

Neighbor rows have no ordering or tie-breaking guarantee. Queries paired with
an empty reference batch receive all-invalid rows.

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

scores = tp.fscore(prediction, reference, threshold=torch.tensor([0.01, 0.05]))
metrics = tp.point_cloud_metrics(prediction, reference, threshold=0.05)
```

Chamfer distances are differentiable with nearest-neighbor identity treated as
piecewise constant. Convert padded batches to `PointCloud` before computing
metrics.

## Registration

Single-scale ICP accepts tensors or packed clouds. Results are always batched:

```python
result = tp.icp(
    source,
    target,
    max_distance=0.1,
    method="point_to_plane",
    target_normals=normals,
    max_iterations=30,
)

result.transforms      # (B, 4, 4), source-to-target
result.converged       # (B,), bool
result.iterations      # (B,), accumulated updates
result.fitness         # (B,), inliers / source length
result.inlier_rmse     # (B,)
```

Correspondences remain internal. Batch entries converge or fail independently,
and failures retain their last valid transform.

Use multi-scale ICP for larger initial misalignment:

```python
scales = [
    tp.ICPScale(voxel_size=0.20, max_distance=0.40, iterations=30),
    tp.ICPScale(voxel_size=0.10, max_distance=0.20, iterations=20),
    tp.ICPScale(voxel_size=0.05, max_distance=0.10, iterations=15),
]
result = tp.multiscale_icp(source, target, scales, method="point_to_plane")
```

Reuse preprocessing by passing `PointCloudPyramid` objects directly:

```python
sizes = [scale.voxel_size for scale in scales]
source_pyramid = tp.build_pyramid(source, sizes)
target_pyramid = tp.build_pyramid(target, sizes, normal_mode="estimate")
result = tp.multiscale_icp(source_pyramid, target_pyramid, scales,
                           method="point_to_plane")
```

`evaluate_registration` evaluates supplied transforms without iteration and
returns `RegistrationMetrics`.

## Behavior

- Geometry supports float32 and float64 on CPU and CUDA.
- Paired inputs must have identical dtype, device, and batch size.
- Procrustes alignment, transforms, voxel reductions, and gathered metric
  distances support autograd. Search indices are discrete; normals and ICP are
  inference-only.
- CPU and CUDA may choose different indices for exact distance ties.
- Empty batch entries are supported by storage, transforms, voxelization, and
  search queries. Metrics and registration reject empty cloud pairs.
- Packed batches currently use exact brute-force search; BVH indexing currently
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
uv run python benchmarks/run_benchmark.py --task all --repeats 5
uv run python benchmarks/run_benchmark.py --task multiscale --repeats 5
uv run python benchmarks/run_benchmark.py --task knn --knn-sizes 512 2048 8192
```

The multi-scale benchmark is end-to-end: it includes voxel pyramids, automatic
point-to-plane normals, neighbor indices, and all ICP levels. Configure its
schedule with `--multiscale-voxels`, `--multiscale-distances`, and
`--multiscale-iterations`; all three lists must have the same length.

Install the `benchmark` dependency group to add small_gicp and Open3D rows where
their Python packages are available. Recorded baseline results are in
[benchmarks/BASELINE.md](benchmarks/BASELINE.md).
