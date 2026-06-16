# torchpcl

Minimal point cloud registration and processing library built on
[PyTorch](https://pytorch.org) (tensor / linear-algebra backend) and
[cuBQL](https://github.com/NVIDIA/cuBQL) (BVH build and queries),
reimplementing the core of Open3D's ICP registration pipeline.

All spatial search (ICP correspondences, k-NN for normals, metric
distances) runs on a cuBQL BVH, on **both CPU and CUDA**: the same
traversal code is compiled as a CUDA extension for GPU tensors and as a
plain C++ extension (parallelized over torch's intra-op thread pool) for
CPU tensors.

For end users:

```bash
pip install torchpcl
```

For the JIT build toolchain used by the cuBQL extensions:

```bash
git submodule update --init third_party/cuBQL
uv sync --group dev
```

Wheel and sdist installs include the extension sources and vendored
cuBQL headers. The CPU extension needs a C++ compiler and ninja. The CUDA extension
additionally needs the nvcc/CCCL toolchain from the dev dependency group,
or a system CUDA toolkit matching `torch.version.cuda`'s major version.
The CUDA build targets only the local GPU architecture; set
`TORCHPCL_CUDA_ARCH_LIST` to override, `TORCHPCL_CUBQL_DIR` to point at
an external cuBQL checkout, or `CUDA_HOME` to choose a toolkit. CPU and
CUDA may tie-break equidistant neighbors differently; compare poses and
metrics across devices, not correspondence indices.

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
```

Points are processed in the input precision (float32 recommended); only
the cumulative transformation and the small per-iteration solves are
float64. The API mirrors `open3d.t.pipelines.registration.icp`
semantics: hybrid correspondence search (nearest target point within
`max_correspondence_distance`), Umeyama solve for point-to-point,
linearized 6-DOF solve for point-to-plane, convergence on relative
fitness/RMSE change.

**Deviation from Open3D:** when no correspondences are found, Open3D
resets the transformation to identity; torchpcl keeps the current
transformation and returns `converged=False, fitness=0`.

## Development

```bash
uv sync --group dev
uv run pytest -q   # first run JIT-compiles the extensions; CUDA tests
                   # skip automatically on machines without a GPU
```

### Benchmark

`benchmarks/run_benchmark.py` registers the sample scans in `data/`
(source/target + ground-truth `T_target_source.txt`) and reports pose
error and wall time. It also benchmarks voxel downsampling and normal
estimation. torchpcl rows run with the default dependencies; install the
benchmark group to add small_gicp and open3d comparison rows. Open3D is
installed by the benchmark group only on Python 3.12, where its wheels
are available; on newer Python versions those rows are skipped.

```bash
uv run python benchmarks/run_benchmark.py [--task all] [--voxel 0.25] [--repeats 5]
uv run --group benchmark python benchmarks/run_benchmark.py
```

Registration inputs are voxel-downsampled and normals are estimated
outside the timed registration loop for each library. Each timed
registration run includes the library's own search-structure build and
the full registration from identity.

### Cross-check against Open3D

`tests/test_open3d_crosscheck.py` compares results against
`open3d.pipelines.registration.registration_icp` on identical inputs and
skips when open3d is not importable. To include it:

```bash
uv sync --group dev --group benchmark
uv run pytest -q tests/test_open3d_crosscheck.py
```
