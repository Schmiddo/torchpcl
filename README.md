# torchpcl

Minimal point cloud registration and processing library built on [PyTorch](https://pytorch.org) and [cuBQL](https://github.com/NVIDIA/cuBQL), inspired by [Open3D's](https://open3d.org) ICP pipeline.

All spatial search (ICP correspondences, k-NN for normals, metric distances) runs on a cuBQL BVH, on **both CPU and CUDA**.

## Installation

Requires a C++ compiler and ninja, as well as the CUDA toolkit for GPU acceleration.
Make sure your CUDA toolkit matches the version used in your PyTorch installation.

The CUDA build targets only the local GPU architecture; set `TORCHPCL_CUDA_ARCH_LIST` to override, `TORCHPCL_CUBQL_DIR` to point at an external cuBQL checkout, or `CUDA_HOME` to choose a toolkit.
CPU and CUDA may tie-break equidistant neighbors differently, so correspondence indices might differ; poses and metrics should be comparable across devices.


## Usage

```python
import torch
import torchpcl as tp

source = torch.randn(10_000, 3, device="cuda")   # (N, 3); CPU works too
target = ...                                     # (M, 3), same device

# Point-to-point (default)
result = tp.icp(source, target, max_correspondence_distance=0.1)

# Point-to-plane (requires target normals)
result = tp.icp(
    source, target, 0.1,
    estimation=tp.PointToPlane(),
    target_normals=normals,                      # (M, 3)
    criteria=tp.ICPConvergenceCriteria(max_iteration=50),
)

result.transformation   # (4, 4) float64 source-to-target
result.fitness          # inliers / N
result.inlier_rmse
result.converged
result.correspondences  # (N,) int64 target index, -1 = none

# Evaluate a given transformation without iterating
tp.evaluate_registration(source, target, 0.1, transformation)

# Preprocessing
down = tp.voxel_downsample(target, voxel_size=0.05)  # per-voxel means
normals = tp.estimate_normals(down, k=30)            # unbounded k-NN + PCA
normals = tp.estimate_normals(down, radius=0.2, k=30, viewpoint=...)

# Cloud comparison (accuracy/completion from prediction->reference /
# reference->prediction; chamfer = accuracy + completion)
m = tp.point_cloud_metrics(prediction, reference, threshold=0.05)
m.accuracy, m.completion, m.chamfer_distance, m.precision, m.recall, m.f1_score

# Differentiable chamfer loss for training (squared L2 by default;
# squared=False matches chamfer_distance above). Also accepts padded
# (B, N, 3) batches with per-cloud lengths.
loss = tp.chamfer_loss(prediction, reference)  # 0-dim; grads flow to both clouds
loss.backward()
```

Points are processed in the input precision (float32 recommended); the cumulative transformation and per-iteration computations are float64.
The API mirrors `open3d.t.pipelines.registration.icp` semantics.
When no correspondences are found, Open3D resets the transformation to identity; torchpcl keeps the current transformation and returns `converged=False, fitness=0`.

## Development

```bash
uv sync --group dev
uv run pytest -q   # first run JIT-compiles the extensions; CUDA tests
                   # skip automatically on machines without a GPU
```

### Benchmark

`benchmarks/run_benchmark.py` registers the sample scans in `data/` (source/target + ground-truth `T_target_source.txt`) and reports pose error and wall time.
It also benchmarks voxel downsampling, normal estimation, and the chamfer loss (against a `torch.cdist` brute force).
Install the benchmark group to compare to small_gicp and open3d.
Open3D is installed by the benchmark group only on Python 3.12, where its wheels are available; on newer Python versions those rows are skipped.

```bash
uv run python benchmarks/run_benchmark.py [--task all] [--voxel 0.25] [--repeats 5]
uv run --group benchmark python benchmarks/run_benchmark.py
```

Registration inputs are voxel-downsampled and normals are estimated outside the timed registration loop for each library.
Each timed registration run includes the library's own search-structure build and the full registration from identity.

### Cross-check against Open3D

`tests/test_open3d_crosscheck.py` compares results against `open3d.pipelines.registration.registration_icp` on identical inputs and skips when open3d is not importable.
To include it:

```bash
uv sync --group dev --group benchmark
uv run pytest -q tests/test_open3d_crosscheck.py
```

