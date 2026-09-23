"""Shared workload definition for the operation-style drivers.

The operations suite measures these workloads and `accuracy.py` checks them for
agreement, so both need the same arguments, the same input preparation and the
same recorded configuration. Keeping that in one place means a parameter cannot
drift between what is timed and what is checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import features_cases as features
from . import neighbors_cases as neighbors
from . import preprocess_cases as preprocess
from .cli import positive_float, positive_int
from .inputs import load_fragment, random_normals, random_points

DEFAULT_FRAGMENT = Path("data/livingroom1-fragments-ply/cloud_bin_0.ply")
# Redwood fragments are metric and far denser than uniform points in a unit
# cube, so radii tuned for one overrun the neighbor cap on the other.
DEFAULT_RADIUS = {"random": 0.02, "redwood": 0.01}
DEFAULT_OUTLIER_RADIUS = {"random": 0.05, "redwood": 0.02}
# Kept under the 64-neighbor cap at the default cloud sizes, so neither
# library summarizes a truncated neighborhood.
DEFAULT_FEATURE_RADIUS = {"random": 0.03, "redwood": 0.015}

ALL_OPS = (*neighbors.OPS, *preprocess.OPS, *features.OPS)


def add_arguments(parser) -> None:
    """Input selection and per-operation parameters, shared by both drivers."""
    parser.add_argument("--normals", type=Path,
                        help="FPFH normals as an (N, 3) .npy array in point order")
    parser.add_argument("--input", choices=["random", "redwood"], default="random",
                        help="uniform points in a unit cube, or a scan fragment")
    parser.add_argument("--fragment", type=Path, default=DEFAULT_FRAGMENT,
                        help="PLY fragment used by --input redwood")
    parser.add_argument("--num-reference", type=positive_int, default=200_000,
                        help="random-input cloud size; a fragment's own size wins")
    parser.add_argument("--num-queries", type=positive_int, default=50_000)
    parser.add_argument("--k", type=positive_int, default=30,
                        help=f"neighbors per query (at most {neighbors.MAX_NEIGHBORS})")
    parser.add_argument("--radius", type=positive_float,
                        help=f"radius-search radius (default: {DEFAULT_RADIUS})")
    parser.add_argument("--voxel-size", type=positive_float, default=0.02)
    parser.add_argument("--normal-k", type=positive_int, default=30)
    parser.add_argument("--outlier-k", type=positive_int, default=20,
                        help="maps directly to Open3D's nb_neighbors / nb_points")
    parser.add_argument("--std-ratio", type=positive_float, default=2.0)
    parser.add_argument("--outlier-radius", type=positive_float,
                        help=f"default: {DEFAULT_OUTLIER_RADIUS}")
    parser.add_argument("--feature-radius", type=positive_float,
                        help=f"FPFH radius (default: {DEFAULT_FEATURE_RADIUS})")
    parser.add_argument("--feature-neighbors", type=positive_int,
                        default=features.MAX_NEIGHBORS,
                        help=f"FPFH neighbors per point (at most "
                             f"{features.MAX_NEIGHBORS})")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=0)


def resolve_defaults(args, parser) -> None:
    """Apply input-dependent defaults and reject out-of-range parameters."""
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    if args.k > neighbors.MAX_NEIGHBORS:
        parser.error(f"--k must be at most {neighbors.MAX_NEIGHBORS}")
    if args.feature_neighbors > features.MAX_NEIGHBORS:
        parser.error(f"--feature-neighbors must be at most {features.MAX_NEIGHBORS}")
    if args.radius is None:
        args.radius = DEFAULT_RADIUS[args.input]
    if args.outlier_radius is None:
        args.outlier_radius = DEFAULT_OUTLIER_RADIUS[args.input]
    if args.feature_radius is None:
        args.feature_radius = DEFAULT_FEATURE_RADIUS[args.input]


@dataclass
class Workload:
    """Prepared inputs for the selected operations, plus how they were built."""

    args: object
    ops: list[str]
    neighbors: neighbors.PreparedNeighbors | None
    cloud: preprocess.PreparedCloud | None
    features: features.PreparedFeatures | None
    normals_source: str | None

    def config(self) -> dict:
        args = self.args
        return {
            "ops": self.ops, "input": args.input,
            "fragment": str(args.fragment) if args.input == "redwood" else None,
            "num_reference": args.num_reference, "num_queries": args.num_queries,
            "k": args.k, "radius": args.radius, "voxel_size": args.voxel_size,
            "normal_k": args.normal_k, "outlier_k": args.outlier_k,
            "std_ratio": args.std_ratio, "outlier_radius": args.outlier_radius,
            "feature_radius": args.feature_radius,
            "feature_neighbors": args.feature_neighbors,
            "dtype": args.dtype, "seed": args.seed,
            "normals_source": self.normals_source,
            "normals": (str(args.normals) if self.features is not None
                        and args.normals is not None else None),
        }

    def input_metadata(self) -> dict:
        prepared = {"neighbors": self.neighbors, "preprocess": self.cloud,
                    "features": self.features}
        return {name: value.metadata() for name, value in prepared.items()
                if value is not None}

    def describe(self) -> list[str]:
        """The lines each driver prints above its table."""
        args, lines = self.args, []
        if self.neighbors is not None:
            info = self.neighbors.metadata()
            line = (f"  search: reference {info['num_reference']}, queries "
                    f"{info['num_queries']}, k {args.k}")
            if "radius" in self.ops:
                # Brute forcing this is what the radius guard already paid for;
                # other operations never compute it.
                line += (f", radius {args.radius}; densest sampled neighborhood "
                         f"{self.neighbors.max_radius_count} (torchpcl returns "
                         f"at most {neighbors.MAX_NEIGHBORS})")
            lines.append(line)
        if self.cloud is not None:
            lines.append(
                f"  preprocess: {self.cloud.metadata()['num_points']} points, "
                f"voxel {args.voxel_size}, normal k {args.normal_k}, "
                f"outlier k {args.outlier_k}")
        if self.features is not None:
            lines.append(
                f"  features: {self.features.metadata()['num_points']} points, "
                f"radius {args.feature_radius}, neighbors "
                f"{args.feature_neighbors}; normals: {self.normals_source}")
        return lines


def prepare(args, ops: list[str]) -> Workload:
    """Build every input the selected operations need, once, from one cloud."""
    fragment, fragment_normals = ((None, None) if args.input == "random"
                                  else load_fragment(args.fragment, args.dtype))
    source = "fragment" if fragment is not None else "random"

    def points_for(count):
        return fragment if fragment is not None else random_points(
            count, args.dtype, args.seed)

    prepared_neighbors = prepared_cloud = prepared_features = None
    if any(op in neighbors.OPS for op in ops):
        prepared_neighbors = neighbors.prepare_neighbors(
            neighbors.NeighborsConfig(
                num_reference=args.num_reference, num_queries=args.num_queries,
                k=args.k, radius=args.radius, dtype=args.dtype, seed=args.seed),
            fragment)
    if any(op in preprocess.OPS for op in ops):
        prepared_cloud = preprocess.prepare_cloud(
            preprocess.PreprocessConfig(
                voxel_size=args.voxel_size, normal_k=args.normal_k,
                outlier_k=args.outlier_k, std_ratio=args.std_ratio,
                outlier_radius=args.outlier_radius, dtype=args.dtype),
            points_for(args.num_reference), source)
    normals_source = None
    if any(op in features.OPS for op in ops):
        points = points_for(args.num_reference)
        if args.normals is not None:
            normals = np.load(args.normals, allow_pickle=False)
            normals_source = "npy"
        elif fragment is not None:
            normals = fragment_normals
            if normals is None:
                raise ValueError(
                    f"{args.fragment}: FPFH needs nx/ny/nz properties "
                    "or --normals PATH.npy")
            normals_source = "ply"
        else:
            normals = random_normals(len(points), args.seed)
            normals_source = "synthetic"
        prepared_features = features.prepare_features(
            features.FeaturesConfig(
                radius=args.feature_radius, max_neighbors=args.feature_neighbors,
                dtype=args.dtype),
            points, normals, source)
    return Workload(args, ops, prepared_neighbors, prepared_cloud,
                    prepared_features, normals_source)


__all__ = ["ALL_OPS", "DEFAULT_FRAGMENT", "Workload", "add_arguments",
           "prepare", "resolve_defaults"]
