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

from dataclasses import dataclass, field

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
# Queries brute forced when an answer is requested. The full cross product is
# far too large to materialize at benchmark sizes, and a fixed prefix is enough
# to catch a backend returning wrong neighbors.
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
    """The shared clouds. Brute-force answers are computed only on request.

    A timing run needs neither: it measures a search, it does not check one.
    The answers cost seconds at benchmark sizes, so they are produced lazily by
    whoever actually wants them -- the accuracy tool, and the radius capacity
    guard below.
    """

    config: NeighborsConfig
    source: str
    reference: np.ndarray
    queries: np.ndarray
    _distances2: np.ndarray | None = field(default=None, repr=False)
    _counts: np.ndarray | None = field(default=None, repr=False)

    @property
    def gate(self) -> int:
        return min(GATE_QUERIES, len(self.queries))

    def _blocks(self):
        """Squared distances from the scored prefix to the whole reference.

        Chunked so the prefix never materializes a full distance matrix; the
        chunk shrinks as the reference grows to hold the peak roughly fixed.
        """
        import torch

        reference = torch.from_numpy(self.reference)
        queries = torch.from_numpy(self.queries[:self.gate])
        chunk = max(1, 64_000_000 // (3 * len(self.reference)))
        for start in range(0, self.gate, chunk):
            block = queries[start:start + chunk]
            yield (block[:, None, :] - reference[None, :, :]).square().sum(-1)

    def knn_distances2(self) -> np.ndarray:
        """Brute-forced k nearest squared distances for the scored prefix."""
        import torch

        if self._distances2 is None:
            self._distances2 = torch.cat(
                [block.topk(self.config.k, dim=1, largest=False).values
                 for block in self._blocks()]).numpy()
        return self._distances2

    def radius_counts(self) -> np.ndarray:
        """Brute-forced neighbor counts within the radius, for the prefix."""
        import torch

        if self._counts is None:
            squared = self.config.radius * self.config.radius
            self._counts = torch.cat(
                [(block <= squared).sum(1) for block in self._blocks()]).numpy()
        return self._counts

    @property
    def max_radius_count(self) -> int:
        return int(self.radius_counts().max())

    def metadata(self):
        return {
            "source": self.source,
            "num_reference": int(self.reference.shape[0]),
            "num_queries": int(self.queries.shape[0]),
            "k": self.config.k,
            "radius": self.config.radius,
        }


def prepare_neighbors(config: NeighborsConfig,
                      points: np.ndarray | None = None) -> PreparedNeighbors:
    """Prepare both clouds. Nothing is brute forced until something asks.

    `points` supplies a loaded cloud (a scan fragment) as the reference;
    queries are then a seeded subsample of it, since searching a cloud against
    itself is what normals and feature extraction actually do. Without it both
    clouds are drawn at random.
    """
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
    return PreparedNeighbors(config=config, source=source,
                             reference=reference, queries=queries)


def check_radius_capacity(prepared: PreparedNeighbors) -> int:
    """Reject a radius that overruns torchpcl's fixed-width result.

    Returns the densest sampled neighborhood. This samples a query prefix: a
    sampled maximum below the cap does not prove that every query is
    untruncated, only that the sampled ones are.
    """
    count = prepared.max_radius_count
    if count > MAX_NEIGHBORS:
        raise ValueError(
            f"radius {prepared.config.radius} yields up to {count} neighbors, "
            f"above torchpcl's cap of {MAX_NEIGHBORS}; reduce --radius so both "
            "backends return complete neighborhoods")
    return count


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


def make_case(prepared: PreparedNeighbors, op: str, backend: str,
              device: str = "cpu") -> Case:
    """Allocate inputs once and return an independently reusable case."""
    if op not in OPS:
        raise ValueError(f"op must be one of {', '.join(OPS)}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if op == "radius":
        check_radius_capacity(prepared)

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

    return Case(
        name=op, backend=backend, device=device,
        run=run, synchronize=synchronize,
        device_memory_reader=device_memory_reader,
        metadata={"input_dtype": prepared.config.dtype,
                  "implementation": implementation},
    )


__all__ = [
    "BACKENDS", "MAX_NEIGHBORS", "OPS", "NeighborsConfig", "PreparedNeighbors",
    "check_radius_capacity", "make_case", "prepare_neighbors",
]
