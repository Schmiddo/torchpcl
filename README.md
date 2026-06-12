# torchpcl

Minimal ICP registration pipeline built on [PyTorch](https://pytorch.org)
(tensor / linear-algebra backend) and [NVIDIA Warp](https://github.com/NVIDIA/warp)
(hash-grid nearest-neighbor search), reimplementing the core of Open3D's
registration pipeline. Runs on CPU and CUDA.

## Usage

```python
import torch
import torchpcl as tp

source = torch.randn(10_000, 3, device="cuda")   # (N, 3)
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
```

The API mirrors `open3d.t.pipelines.registration.icp` semantics:
hybrid correspondence search (nearest target point within
`max_correspondence_distance`), Umeyama solve for point-to-point,
linearized 6-DOF solve for point-to-plane, convergence on relative
fitness/RMSE change.

**Deviation from Open3D:** when no correspondences are found, Open3D
resets the transformation to identity; torchpcl keeps the current
transformation and returns `converged=False, fitness=0`.

## Development

```bash
uv sync
uv run pytest -q
```

Tests run on CPU and, when available, CUDA.

### Benchmark

`benchmarks/run_benchmark.py` registers the sample scans in `data/`
(source/target + ground-truth `T_target_source.txt`) and reports pose
error and wall time for torchpcl (CPU and CUDA), small_gicp, and open3d
when importable:

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
VIRTUAL_ENV=/tmp/o3d-venv uv pip install torch warp-lang pytest open3d
PYTHONPATH=src /tmp/o3d-venv/bin/python -m pytest -q tests/
```
