# torchpcl

Minimal point cloud registration and processing library built on
[PyTorch](https://pytorch.org) (tensor / linear-algebra backend) and
[cuBQL](https://github.com/NVIDIA/cuBQL) (GPU BVH build and queries),
reimplementing the core of Open3D's ICP registration pipeline.

All spatial search (ICP correspondences, k-NN for normals, metric
distances) runs on a cuBQL BVH, on **both CPU and CUDA**: the same
traversal code is compiled as a CUDA extension for GPU tensors and as a
plain C++ extension (parallelized over torch's intra-op thread pool)
for CPU tensors. Extensions are JIT-compiled on first use (cached
afterwards), so torchpcl requires a dev checkout with
`third_party/cuBQL`:

```bash
git submodule update --init third_party/cuBQL
uv sync   # installs torch plus the pip nvcc/CCCL toolchain (pinned to torch's CUDA minor)
```

The CPU extension needs only a C++ compiler and ninja; the CUDA
extension additionally needs the nvcc toolchain installed by `uv sync`.
The CUDA build targets only the local GPU architecture; set
`TORCHPCL_CUDA_ARCH_LIST` to override, `TORCHPCL_CUBQL_DIR` to point at
an external cuBQL checkout, or `CUDA_HOME` at a system toolkit matching
`torch.version.cuda`'s major version. CPU and CUDA may tie-break
equidistant neighbors differently; compare poses and metrics across
devices, not correspondence indices.

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
# backend="torch": exact chunked brute force reference implementation
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
uv sync
uv run pytest -q   # first run JIT-compiles the extensions; CUDA tests
                   # skip automatically on machines without a GPU
```

### Benchmark

`benchmarks/run_benchmark.py` registers the sample scans in `data/`
(source/target + ground-truth `T_target_source.txt`) and reports pose
error and wall time for torchpcl, small_gicp, and open3d when
importable:

```bash
uv run python benchmarks/run_benchmark.py [--voxel 0.25] [--repeats 5]
```

Both clouds are voxel-downsampled and normals are estimated once,
shared by all methods; each timed run includes the library's own
search-structure build and the full registration from identity.

### Cross-check against Open3D

`tests/test_open3d_crosscheck.py` compares results against
`open3d.pipelines.registration.registration_icp` on identical inputs and
skips when open3d is not importable. open3d has no Python 3.14 wheels
yet, so run it from a Python ≤3.12 environment, e.g.:

```bash
uv venv -p 3.12 /tmp/o3d-venv
VIRTUAL_ENV=/tmp/o3d-venv uv pip install torch ninja open3d numpy \
    "nvidia-cuda-nvcc~=13.0.0" "nvidia-cuda-cccl~=13.0.0" \
    "nvidia-cuda-runtime~=13.0.0" "nvidia-cuda-crt~=13.0.0" "nvidia-nvvm~=13.0.0"
PYTHONPATH=src /tmp/o3d-venv/bin/python -m pytest -q tests/
```
