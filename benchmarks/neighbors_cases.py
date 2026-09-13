"""Prepare shared clouds and construct neighbor-search cases per backend.

Covers the three scopes separately, because they answer different questions:
building an index, querying k nearest neighbors, and querying a fixed radius.
torchpcl builds one `NeighborIndex` that serves every search type, while
Open3D builds a separate index per type (`knn_index`, `fixed_radius_index`);
the build case times each library's own path to a queryable knn index.

The libraries do not split that work the same way. Open3D's CUDA `knn_index`
returns in microseconds however large the reference cloud is, and its CUDA
query is correspondingly dearer, so `index-build` and `knn` are not comparable
in isolation on CUDA. `knn-build-query` times both together, from raw points,
which is the total either library actually charges for answering a batch of
queries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .case import Case
from .inputs import random_points
from .memory import cuda_process_memory_reader

OPS = ("index-build", "knn", "knn-build-query", "radius")
BACKENDS = ("torchpcl", "open3d")
# torchpcl's search API caps neighbors per query; a radius holding more true
# neighbors than this would silently compare a truncated result against a
# complete one.
MAX_NEIGHBORS = 64
# Queries scored against brute force. The full cross product is far too large
# to materialize at benchmark sizes, and a fixed prefix is enough to catch a
# backend returning wrong neighbors.
GATE_QUERIES = 1024


@dataclass(frozen=True)
class NeighborsConfig:
    num_reference: int = 200_000
    num_queries: int = 50_000
    k: int = 30
    radius: float = 0.02
    dtype: str = "float32"
    seed: int = 0

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        for name in ("num_reference", "num_queries"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.k, int) or not 1 <= self.k <= MAX_NEIGHBORS:
            raise ValueError(f"k must be an integer in [1, {MAX_NEIGHBORS}]")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")


@dataclass
class PreparedNeighbors:
    """Shared clouds plus the brute-force answers the cases are scored against."""

    config: NeighborsConfig
    source: str
    reference: np.ndarray
    queries: np.ndarray
    gate_distances2: np.ndarray
    gate_counts: np.ndarray
    max_radius_count: int

    def metadata(self):
        return {
            "source": self.source,
            "num_reference": int(self.reference.shape[0]),
            "num_queries": int(self.queries.shape[0]),
            "k": self.config.k,
            "radius": self.config.radius,
            "gate_queries": int(self.gate_distances2.shape[0]),
            # Truncation would invalidate the radius comparison, so the count
            # that drove the check is reported rather than merely asserted on.
            "max_radius_count": self.max_radius_count,
        }


def prepare_neighbors(config: NeighborsConfig,
                      points: np.ndarray | None = None) -> PreparedNeighbors:
    """Prepare both clouds once and brute-force the scored query prefix.

    `points` supplies a loaded cloud (a scan fragment) as the reference;
    queries are then a seeded subsample of it, since searching a cloud against
    itself is what normals and feature extraction actually do. Without it both
    clouds are drawn at random.
    """
    import torch

    if points is None:
        source = "random"
        reference = random_points(config.num_reference, config.dtype, config.seed)
        queries = random_points(config.num_queries, config.dtype, config.seed + 1)
    else:
        source = "fragment"
        reference = points
        rng = np.random.default_rng(config.seed + 1)
        count = min(config.num_queries, len(reference))
        queries = reference[rng.choice(len(reference), count, replace=False)]

    gate = min(GATE_QUERIES, len(queries))
    reference_t = torch.from_numpy(reference)
    gate_t = torch.from_numpy(queries[:gate])
    # Chunked so the scored prefix never materializes a full distance matrix;
    # the chunk shrinks as the reference grows to hold the peak roughly fixed.
    chunk = max(1, 64_000_000 // (3 * len(reference)))
    distances2, counts = [], []
    for start in range(0, gate, chunk):
        block = gate_t[start:start + chunk]
        block_d2 = (block[:, None, :] - reference_t[None, :, :]).square().sum(-1)
        distances2.append(block_d2.topk(config.k, dim=1, largest=False).values)
        counts.append((block_d2 <= config.radius * config.radius).sum(1))
    gate_counts = torch.cat(counts)

    return PreparedNeighbors(
        config=config,
        source=source,
        reference=reference,
        queries=queries,
        gate_distances2=torch.cat(distances2).numpy(),
        gate_counts=gate_counts.numpy(),
        max_radius_count=int(gate_counts.max()),
    )


def _torchpcl_cases(prepared, op, device):
    import torch
    import torchpcl as tp

    config = prepared.config
    dtype = getattr(torch, config.dtype)
    reference = torch.tensor(prepared.reference, dtype=dtype, device=device)
    queries = torch.tensor(prepared.queries, dtype=dtype, device=device)

    if op == "index-build":
        return lambda: tp.NeighborIndex(reference), "bvh"
    if op == "knn-build-query":
        return (
            lambda: tp.NeighborIndex(reference).knn(queries, config.k), "bvh")

    # Only the query is timed, so the index is built here, during preparation.
    index = tp.NeighborIndex(reference)
    if op == "knn":
        return lambda: index.knn(queries, config.k), index.algorithm
    return (
        lambda: index.radius(queries, config.radius, max_neighbors=MAX_NEIGHBORS),
        index.algorithm,
    )


def _open3d_cases(prepared, op, device):
    import open3d.core as o3c

    config = prepared.config
    dtype = o3c.float32 if config.dtype == "float32" else o3c.float64
    o3d_device = o3c.Device("CUDA:0" if device == "cuda" else "CPU:0")
    reference = o3c.Tensor(
        prepared.reference.astype(config.dtype), dtype, o3d_device)
    queries = o3c.Tensor(
        prepared.queries.astype(config.dtype), dtype, o3d_device)

    if op == "index-build":
        def build():
            # The constructor is lazy; `knn_index` is where the build happens,
            # so both belong inside the timed region.
            index = o3c.nns.NearestNeighborSearch(reference)
            index.knn_index()
            return index

        return build, "open3d-nns"

    if op == "knn-build-query":
        def build_and_query():
            index = o3c.nns.NearestNeighborSearch(reference)
            index.knn_index()
            return index.knn_search(queries, config.k)

        return build_and_query, "open3d-nns"

    index = o3c.nns.NearestNeighborSearch(reference)
    if op == "knn":
        index.knn_index()
        return lambda: index.knn_search(queries, config.k), "open3d-nns"
    index.fixed_radius_index(config.radius)
    return (
        lambda: index.fixed_radius_search(queries, config.radius),
        "open3d-nns",
    )


def _make_evaluate(prepared, op, backend):
    """Score a run against the brute-force prefix, or nothing to score."""
    if op == "index-build":
        return None
    gate = prepared.gate_distances2.shape[0]

    if op in ("knn", "knn-build-query"):
        expected = np.sort(prepared.gate_distances2, axis=1)

        def evaluate(result):
            if backend == "torchpcl":
                found = result.distances2[:gate].double().cpu().numpy()
            else:
                found = result[1][:gate].cpu().numpy().astype(np.float64)
            # Neighbor order is unspecified on both sides; only the multiset
            # of distances is contractual.
            error = np.abs(np.sort(found, axis=1) - expected).max()
            return {"max_distance2_error": float(error)}

        return evaluate

    expected_counts = prepared.gate_counts

    def evaluate(result):
        if backend == "torchpcl":
            found = result.valid[:gate].sum(1).cpu().numpy()
        else:
            # Ragged: flat neighbors plus per-query row splits.
            found = np.diff(result[2].cpu().numpy())[:gate]
        return {
            "count_mismatch_fraction": float(
                np.mean(found != expected_counts)),
        }

    return evaluate


def make_case(prepared: PreparedNeighbors, op: str, backend: str,
              device: str = "cpu") -> Case:
    """Allocate inputs once and return an independently reusable case."""
    if op not in OPS:
        raise ValueError(f"op must be one of {', '.join(OPS)}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if op == "radius" and prepared.max_radius_count > MAX_NEIGHBORS:
        raise ValueError(
            f"radius {prepared.config.radius} yields up to "
            f"{prepared.max_radius_count} neighbors, above torchpcl's cap of "
            f"{MAX_NEIGHBORS}; reduce --radius so both backends return "
            "complete neighborhoods")

    def synchronize():
        pass

    device_memory_reader = None
    if device == "cuda":
        device_memory_reader = cuda_process_memory_reader()

    if backend == "torchpcl":
        import torch

        run, implementation = _torchpcl_cases(prepared, op, device)
        if device == "cuda":
            synchronize = torch.cuda.synchronize
    else:
        import open3d.core as o3c

        run, implementation = _open3d_cases(prepared, op, device)
        if device == "cuda":
            synchronize = o3c.cuda.synchronize

    evaluate = _make_evaluate(prepared, op, backend)
    return Case(
        name=op, backend=backend, device=device,
        run=run, synchronize=synchronize,
        device_memory_reader=device_memory_reader,
        **({} if evaluate is None else {"evaluate": evaluate}),
        metadata={"input_dtype": prepared.config.dtype,
                  "implementation": implementation},
    )


__all__ = [
    "BACKENDS", "MAX_NEIGHBORS", "OPS", "NeighborsConfig", "PreparedNeighbors",
    "make_case", "prepare_neighbors",
]
