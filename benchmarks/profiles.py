"""Named, pinned workloads: the only runs a history keeps.

A profile fixes every argument that defines a suite's work, so two runs of it
measure the same thing on any revision. What it leaves open only selects which
part of that work runs -- backend, device, repeats, an operation subset -- and
never changes a number that is measured.

A history series is keyed on the profile's name and version, not on a digest
of whatever configuration a report happened to record: adding a report field
cannot split a series, and changing the workload on purpose is explicit.
**Bump `version` whenever a pinned value changes**; `history.py add` refuses a
report whose pinned values differ from what an earlier point of the same
version recorded. Adding a newly introduced argument at the value it always
implicitly had does not need a bump.

Pure stdlib, so `history.py` and `bench.py` import it without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Pinned rather than derived from the core count, which moved series apart
#: whenever detection changed. Timings compare only within one machine, so a
#: fixed count costs nothing across machines.
THREADS = 8

FRAGMENT = "data/livingroom1-fragments-ply/cloud_bin_0.ply"


@dataclass(frozen=True)
class Profile:
    name: str
    suite: str
    version: int
    params: dict
    #: Files the workload reads; a sweep skips the profile when any is missing.
    requires: tuple[str, ...] = ()
    #: Part of the per-revision sweep. Off for workloads too slow to repeat
    #: on every commit, which are still run and kept explicitly.
    routine: bool = True
    description: str = field(default="", compare=False)

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"

    def missing(self, root: Path = Path(".")) -> list[str]:
        return [path for path in self.requires if not (root / path).exists()]


def _operations(input_kind: str, **radii) -> dict:
    return {
        "input": input_kind, "fragment": FRAGMENT, "normals": None,
        "num_reference": 200_000, "num_queries": 50_000, "k": 30,
        "voxel_size": 0.02, "normal_k": 30, "outlier_k": 20, "std_ratio": 2.0,
        "feature_neighbors": 64, "dtype": "float32", "seed": 0,
        "threads": THREADS, "warmup": 1, **radii,
    }


PROFILES = {profile.name: profile for profile in (
    Profile(
        "operations-random", "operations", 1,
        _operations("random", radius=0.02, outlier_radius=0.05, feature_radius=0.03),
        description="neighbor search, preprocessing and FPFH on 200k uniform points"),
    Profile(
        "operations-redwood", "operations", 1,
        _operations("redwood", radius=0.01, outlier_radius=0.02, feature_radius=0.015),
        requires=(FRAGMENT,),
        description="the same operations on a Redwood scan fragment"),
    Profile(
        "sampling", "sampling", 1,
        {"num_points": 200_000, "num_samples": 5_000, "dtype": "float32", "seed": 0,
         "threads": THREADS, "warmup": 1},
        description="farthest point sampling, 5k of 200k points"),
    Profile(
        "registration", "registration", 1,
        {"data": ["data"], "objective": "point-to-point", "dtype": "float32",
         "voxel_size": 0.05, "icp_distance": None, "fgr_distance": None,
         "icp_iterations": 50, "fgr_iterations": 64, "seed": 0,
         "threads": THREADS, "warmup": 1},
        requires=("data/source.ply", "data/target.ply", "data/T_target_source.txt"),
        description="point-to-point ICP and FGR on the local ground-truth pair"),
    Profile(
        "redwood", "redwood", 1,
        {"data": "data", "scenes": ["livingroom1", "livingroom2", "office1", "office2"],
         "dtype": "float32", "voxel_size": 0.05, "fgr_distance": None,
         "fgr_iterations": 64, "seed": 0, "max_pairs": None,
         "threads": THREADS, "warmup": 1},
        requires=tuple(f"data/{scene}-{kind}" for scene in
                       ("livingroom1", "livingroom2", "office1", "office2")
                       for kind in ("evaluation", "fragments-ply")),
        routine=False,
        description="FGR recall on all 739 Redwood ground-truth pairs"),
)}


def normalize(value):
    """Compare a parsed argument with a pinned value: paths as text, sequences as lists."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


__all__ = ["PROFILES", "Profile", "THREADS", "normalize"]
