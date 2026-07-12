# Pre-refactor Baseline

This baseline records the implementation before the packed-batch refactor. It
is a diagnostic reference, not a release gate: simpler implementations may
accept modest regressions when they substantially reduce maintenance cost.

## Environment

- Date: 2026-07-12
- CPU: Intel Core i9-11900K, 8 cores / 16 threads
- Python: 3.14.3
- PyTorch: 2.12.0+cu130
- CUDA: unavailable on this host; CUDA measurements remain to be collected
- Input: repository sample scans, 69,792 source and 69,088 target points
- Command: `uv run python benchmarks/run_benchmark.py --task all --repeats 3`
- Settings: voxel size 0.25, normal k=30, max correspondence distance 1.0,
  maximum ICP iterations 50

Times are medians of three measured runs after one warmup.

## Results

| Task | Method | CPU time | Detail |
| --- | --- | ---: | --- |
| Voxelization | `voxel_downsample` | 1.52 ms | 69,088 to 6,147 points |
| Normals | `estimate_normals` | 4.27 ms | 6,147 points, k=30 |
| ICP | point-to-point | 8.13 ms | 15 iterations, converged |
| ICP | point-to-plane | 8.06 ms | 12 iterations, converged |
| Chamfer | `torchpcl.chamfer_loss` forward+backward | 1.98 ms | 6,167 vs 6,147 points |
| Chamfer | `torch.cdist` reference forward+backward | 185.61 ms | 6,167 vs 6,147 points |
| k-NN | BVH build, N=512 | 0.05 ms | M=512, k=30 |
| k-NN | BVH query, N=512 | 0.30 ms | M=512, k=30 |
| k-NN | brute-force build, N=512 | 0.01 ms | M=512, k=30 |
| k-NN | brute-force query, N=512 | 0.33 ms | M=512, k=30 |
| k-NN | BVH build, N=2,048 | 0.26 ms | M=1,024, k=30 |
| k-NN | BVH query, N=2,048 | 0.37 ms | M=1,024, k=30 |
| k-NN | brute-force build, N=2,048 | 0.01 ms | M=1,024, k=30 |
| k-NN | brute-force query, N=2,048 | 1.17 ms | M=1,024, k=30 |
| k-NN | BVH build, N=8,192 | 1.15 ms | M=1,024, k=30 |
| k-NN | BVH query, N=8,192 | 0.45 ms | M=1,024, k=30 |
| k-NN | brute-force build, N=8,192 | <0.01 ms | M=1,024, k=30 |
| k-NN | brute-force query, N=8,192 | 2.23 ms | M=1,024, k=30 |

The test baseline at the same revision was 50 passed and one optional Open3D
test skipped. Comparison libraries were not installed, so their benchmark rows
were omitted.

## Phases 2-5 Comparison

After unified search, packed voxelization, normal estimation, and metrics were
implemented, the same CPU preprocessing command reported:

| Task | Baseline | Phases 2-5 | Change |
| --- | ---: | ---: | ---: |
| Voxelization | 1.52 ms | 3.12 ms | 2.05x |
| Normal estimation | 4.27 ms | 9.73 ms | 2.28x |

The slower paths remove fixed-width coordinate packing, add reusable batch
metadata, recompute search distances in the input dtype, and use the shared
`torch.linalg.eigh` implementation. These absolute costs are accepted for the
initial maintainable implementation and remain candidates for profiling after
batched ICP is complete.

## Phase 6 ICP Comparison

The correspondence-free batched ICP implementation reports the following CPU
times on the same data and settings:

| Method | Baseline | Batched ICP | Change |
| --- | ---: | ---: | ---: |
| Point-to-point | 8.13 ms | 15.50 ms | 1.91x |
| Point-to-plane | 8.06 ms | 13.13 ms | 1.63x |

The new path keeps per-batch state and solves on the input device. It reads one
aggregate active flag per iteration to stop after all entries finish; without
that check, a fixed 50-iteration benchmark took approximately 61-63 ms.
