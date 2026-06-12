"""Radius-bounded nearest-neighbor search on a warp hash grid.

This is the only module that imports warp. Everything crossing the
boundary is float32; callers gather from their own float64 copies using
the returned indices.
"""

import torch
import warp as wp

wp.init()

# Hash grid resolution per axis. The warp-example default; adequate for
# v1 but tunable if cells become too coarse for very large scenes.
_GRID_DIM = 128


@wp.kernel
def _knn_kernel(
    grid: wp.uint64,
    queries: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    radius: float,
    k: int,
    out_index: wp.array2d(dtype=wp.int32),
    out_dist2: wp.array2d(dtype=wp.float32),
):
    # Up-to-k nearest within radius, kept sorted by insertion into the
    # output rows (k is small; global-memory insertion sort is fine for
    # a one-shot preprocessing query).
    tid = wp.tid()
    q = queries[tid]
    r2 = radius * radius
    count = int(0)
    for i in wp.hash_grid_query(grid, q, radius):
        d2 = wp.length_sq(points[i] - q)
        if d2 <= r2:
            j = int(0)
            if count < k:
                j = count
                count += 1
            elif d2 < out_dist2[tid, k - 1]:
                j = k - 1
            else:
                continue
            out_index[tid, j] = i
            out_dist2[tid, j] = d2
            while j > 0:
                if out_dist2[tid, j - 1] <= out_dist2[tid, j]:
                    break
                tmp_d = out_dist2[tid, j - 1]
                out_dist2[tid, j - 1] = out_dist2[tid, j]
                out_dist2[tid, j] = tmp_d
                tmp_i = out_index[tid, j - 1]
                out_index[tid, j - 1] = out_index[tid, j]
                out_index[tid, j] = tmp_i
                j -= 1


@wp.kernel
def _nn_kernel(
    grid: wp.uint64,
    queries: wp.array(dtype=wp.vec3),
    points: wp.array(dtype=wp.vec3),
    radius: float,
    out_index: wp.array(dtype=wp.int32),
    out_dist2: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    q = queries[tid]
    best_i = int(-1)
    best_d2 = radius * radius
    for i in wp.hash_grid_query(grid, q, radius):
        d2 = wp.length_sq(points[i] - q)
        if d2 <= best_d2:
            best_d2 = d2
            best_i = i
    out_index[tid] = best_i
    out_dist2[tid] = best_d2


class NearestNeighborSearch:
    """1-NN search within a fixed radius over a static point set.

    Matches Open3D's hybrid search semantics: for each query point, the
    nearest point within ``radius``, or -1 if none exists.
    """

    def __init__(self, points: torch.Tensor, radius: float):
        self._device = str(points.device)
        self._radius = float(radius)
        # Keep a reference so the zero-copy warp view stays valid.
        self._points_f32 = points.to(torch.float32).contiguous()
        self._wp_points = wp.from_torch(self._points_f32, dtype=wp.vec3)
        self._grid = wp.HashGrid(_GRID_DIM, _GRID_DIM, _GRID_DIM, device=self._device)
        with self._scoped_stream():
            self._grid.build(self._wp_points, self._radius)

    def _scoped_stream(self):
        if self._device.startswith("cuda"):
            return wp.ScopedStream(wp.stream_from_torch())
        return wp.ScopedDevice(self._device)

    def query(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (indices, dist2) for each query point.

        indices: (N,) int64, -1 where no point lies within the radius.
        dist2: (N,) float32 squared distance; only meaningful where
            the index is >= 0.
        """
        queries_f32 = queries.to(torch.float32).contiguous()
        out_index = torch.empty(len(queries_f32), dtype=torch.int32, device=queries.device)
        out_dist2 = torch.empty(len(queries_f32), dtype=torch.float32, device=queries.device)
        with self._scoped_stream():
            wp.launch(
                _nn_kernel,
                dim=len(queries_f32),
                inputs=[
                    self._grid.id,
                    wp.from_torch(queries_f32, dtype=wp.vec3),
                    self._wp_points,
                    self._radius,
                    wp.from_torch(out_index, dtype=wp.int32),
                    wp.from_torch(out_dist2, dtype=wp.float32),
                ],
                device=self._device,
            )
        return out_index.to(torch.int64), out_dist2

    def knn_query(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return up to k nearest neighbors within the radius per query.

        indices: (M, k) int64 sorted by distance, -1 padded.
        dist2: (M, k) float32 squared distances; only meaningful where
            the index is >= 0.
        """
        queries_f32 = queries.to(torch.float32).contiguous()
        m = len(queries_f32)
        out_index = torch.full((m, k), -1, dtype=torch.int32, device=queries.device)
        out_dist2 = torch.empty((m, k), dtype=torch.float32, device=queries.device)
        with self._scoped_stream():
            wp.launch(
                _knn_kernel,
                dim=m,
                inputs=[
                    self._grid.id,
                    wp.from_torch(queries_f32, dtype=wp.vec3),
                    self._wp_points,
                    self._radius,
                    k,
                    wp.from_torch(out_index, dtype=wp.int32),
                    wp.from_torch(out_dist2, dtype=wp.float32),
                ],
                device=self._device,
            )
        return out_index.to(torch.int64), out_dist2
