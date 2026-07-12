# torchpcl

Tensor-first point-cloud processing for PyTorch, with exact CPU and CUDA
neighbor search and packed ragged batching.

The focused public API covers neighbor search, voxelization, normal estimation,
point-cloud metrics, and single- or multi-scale ICP. See the [API reference](docs/api.md),
[batching guide](docs/batching.md), and [migration guide](MIGRATION.md).

## Installation

Extensions are compiled at installation time against the PyTorch in the active
environment. Build isolation must be disabled:

```bash
pip install --no-build-isolation .
```

A CUDA build requires `CUDA_HOME` to point to a toolkit compatible with the
installed PyTorch. For a CPU-only build:

```bash
TORCHPCL_CPU_ONLY=1 pip install --no-build-isolation .
```

Set `TORCHPCL_CUDA_ARCH_LIST` to override local GPU architecture detection or
`TORCHPCL_CUBQL_DIR` to use an external cuBQL checkout. Importing torchpcl never
compiles code at runtime.

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

normal_result = tp.estimate_normals(
    downsampled,
    radius=0.15,
    k=30,
    viewpoint=torch.tensor([0.0, 0.0, 0.0], device=points.device),
)
normals = normal_result.normals
valid_normals = normal_result.valid
```

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
- Search indices are discrete; gathered distances and Chamfer metrics support
  autograd. Normals and registration are inference-only.
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
