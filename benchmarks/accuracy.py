"""Check torchpcl's operations against an accepted baseline, without timing.

A benchmark answers "how fast"; this answers "does it still agree". They are
separate commands because combining them makes each worse: a scoring pass runs
on measured output, correctness failures surface as odd timings, and every
performance run drags a reference computation behind it.

Run it deliberately, in the environment that has Open3D. Nothing runs it
automatically, and no test imports it.

Two baselines are used, because they are not equally strong:

- Neighbor searches are checked against **brute force**, the true answer,
  already computed during input preparation. Open3D would be the weaker oracle
  here.
- Everything else is checked against **Open3D on CPU**, the accepted
  implementation. Agreement means the two libraries compute the same thing; a
  check above tolerance means they disagree and someone must look, not that
  torchpcl is necessarily wrong.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

# Preserve direct-script usage as well as python -m benchmarks.accuracy.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import workload
from benchmarks.case import select_devices
from benchmarks.cli import (
    configure_threads,
    default_threads,
    environment_metadata,
    positive_int,
    require_open3d,
    write_json,
)
from benchmarks import features_cases as features
from benchmarks import neighbors_cases as neighbors
from benchmarks import preprocess_cases as preprocess

# Operations whose output can be compared. `index-build` returns an index,
# `knn-build-query` the same neighbors as `knn`, and bare `voxelize` no
# centroids, so none of the three adds a check the others do not already make.
CHECKABLE_OPS = ("knn", "radius", "voxelize-reduce", "normals",
                 "statistical-outliers", "radius-outliers", "fpfh")
OUTLIER_OPS = ("statistical-outliers", "radius-outliers")
# How far the outlier mapping must be from agreeing at the wrong k. Measured
# disagreement at k+-1 is 0.006-0.026, so this floor is well clear of noise
# while still failing loudly if the comparison stops discriminating.
MAPPING_FLOOR = 1e-3
# FPFH bins an angular feature, so a value landing on a bin edge can round to
# either side in float32 and move one neighbor's contribution -- about 100/k of
# the 600 per row -- between two adjacent bins. Verified 2026-09-17 at 200k
# random points: 2 rows differed, each a clean transfer between bins 29 and 30
# with mass conserved, none near the radius boundary and none truncated. Such
# flips grow with cloud size, so the tolerance is a fraction, set an order of
# magnitude above the observed 1e-5; a systematic divergence would affect far
# more rows than this.
DESCRIPTOR_ROW_TOLERANCE = 1e-4
# Both filters threshold a statistic accumulated over the whole cloud (mean and
# standard deviation of neighbor distances), and summation order differs between
# implementations, so the thresholds themselves differ by more than a few ULP.
# Points within that margin fall either way. Verified 2026-09-17 on a 196k-point
# Redwood fragment: 2 points differed, each ~300 float32 ULP from the threshold.
# This stays far below a mapping error, which moves 1e-3 to 6e-2 of all points
# -- the offset check below is what keeps that distinguishable.
MASK_TOLERANCE = 1e-4
# The benchmark's random default (0.05) is past a cliff: at 200k uniform points
# it removes 6 of them, so both libraries agree on "keep everything" and the
# comparison proves nothing. Uniform points in a unit cube have a narrow
# neighbor-count distribution, which makes that transition abrupt -- 0.03
# removes 25% and moves ~12k points when k changes by one. Real fragments
# discriminate at their own default. Timing defaults are deliberately not
# changed here: they would invalidate saved baselines, and a degenerate
# workload is a weaker problem for timing than for agreement.
OUTLIER_RADIUS = {"random": 0.03, "redwood": 0.02}


@dataclass(frozen=True)
class Check:
    """One accuracy claim, its measured value, and the bound it must satisfy."""

    op: str
    device: str
    metric: str
    value: float
    bound: str
    limit: float | None
    reference: str
    reason: str

    @property
    def passed(self) -> bool:
        """Context rows (``bound="info"``) report a number without judging it."""
        if self.bound == "info":
            return True
        if self.bound == "max":
            return self.value <= self.limit
        return self.value >= self.limit

    def row(self) -> dict:
        return {"op": self.op, "device": self.device, "metric": self.metric,
                "value": self.value, "bound": self.bound, "limit": self.limit,
                "reference": self.reference, "reason": self.reason,
                "passed": self.passed}


# --- comparisons -----------------------------------------------------------
# Pure array functions: no Open3D, no torch, so they are testable directly.

def distance_error(found: np.ndarray, expected: np.ndarray) -> float:
    """Largest squared-distance error, comparing each row as a sorted multiset.

    Neighbor order and distance-tie resolution are unspecified on both sides,
    so only the multiset of distances is contractual.
    """
    return float(np.abs(np.sort(found, axis=1) - np.sort(expected, axis=1)).max())


def count_mismatch(found: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean(found != expected))


# Far above float32 centroid noise (order 1e-7) and far below any useful voxel
# size, so it identifies voxels without merging neighbors.
CENTROID_TOLERANCE = 1e-5


def centroid_keys(points: np.ndarray) -> set:
    """Tolerance-quantized centroid identities, comparable as a set."""
    return set(map(tuple, np.round(points / CENTROID_TOLERANCE).astype(np.int64)))


def centroid_mismatch(found: set, expected: set) -> float:
    """Fraction of reference voxels the candidate did not reproduce.

    Set membership rather than sorted rows: one differing voxel would shift
    every row after it and read as a large coordinate error.
    """
    return len(expected - found) / len(expected)


def normal_deviation(found: np.ndarray, expected: np.ndarray) -> float:
    """Worst unsigned misalignment, as ``1 - min|cos|``.

    torchpcl's normal signs are arbitrary without attached normals, so only the
    unsigned direction is contractual.
    """
    return float(1.0 - np.abs((found * expected).sum(1)).min())


def mask_mismatch(found: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean(found != expected))


def descriptor_mismatch(found: np.ndarray, expected: np.ndarray,
                        valid: np.ndarray) -> tuple[float, float]:
    """Fraction of rows disagreeing by more than one bin, and the worst bin.

    Rows are counted rather than a global max taken: descriptors sum to 600 per
    row, and a max over N x 33 values is dominated by single differing
    neighborhoods. Open3D's own CPU and CUDA results differ by several units
    under a max.
    """
    if not valid.any():
        return 0.0, 0.0
    difference = np.abs(found[valid] - expected[valid])
    return (float(np.mean(difference.max(1) > features.BIN_TOLERANCE)),
            float(difference.max()))


# --- running one operation's output ----------------------------------------

def _run(prepared, op, backend, device):
    """Build the same case the benchmark measures, and run it once, untimed."""
    if op in neighbors.OPS:
        case = neighbors.make_case(prepared.neighbors, op, backend, device)
    elif op in preprocess.OPS:
        case = preprocess.make_case(prepared.cloud, op, backend, device)
    else:
        case = features.make_case(prepared.features, op, backend, device)
    return case.run_synchronized()


def _torchpcl_output(op, result, gate):
    if op == "knn":
        return result.distances2[:gate].double().cpu().numpy()
    if op == "radius":
        return result.valid[:gate].sum(1).cpu().numpy()
    if op == "voxelize-reduce":
        return centroid_keys(result.cpu().numpy().astype(np.float64))
    if op == "normals":
        return result.normals.cpu().numpy().astype(np.float64)
    if op in OUTLIER_OPS:
        return ~result.cpu().numpy()
    return (result.descriptors.cpu().numpy().astype(np.float64),
            result.valid.cpu().numpy(),
            result.neighbor_limit_reached.cpu().numpy())


def _open3d_output(op, result):
    if op == "voxelize-reduce":
        return centroid_keys(
            result.point.positions.cpu().numpy().astype(np.float64))
    if op == "normals":
        return result.point.normals.cpu().numpy().astype(np.float64)
    if op in OUTLIER_OPS:
        return result[1].cpu().numpy().astype(bool)
    return result.cpu().numpy().astype(np.float64)


def reference_output(prepared, op):
    """The accepted answer for one operation, computed once on CPU."""
    if op in ("knn", "radius"):
        gate = prepared.neighbors.gate_distances2.shape[0]
        return ("brute-force",
                prepared.neighbors.gate_distances2 if op == "knn"
                else prepared.neighbors.gate_counts, gate)
    return "open3d-cpu", _open3d_output(op, _run(prepared, op, "open3d", "cpu")), None


# --- checks ----------------------------------------------------------------

def check(prepared, op, device, reference) -> list[Check]:
    """Compare one operation's torchpcl output on `device` with the reference."""
    source, expected, gate = reference
    found = _torchpcl_output(op, _run(prepared, op, "torchpcl", device), gate)

    def make(metric, value, reason, *, bound="max", limit=0.0):
        return Check(op, device, metric, value, bound,
                     None if bound == "info" else limit, source, reason)

    if op == "knn":
        return [make("max_distance2_error", distance_error(found, expected),
                     "torchpcl selects candidates in float32 for both input "
                     "dtypes, so exact agreement is not expected", limit=1e-6)]
    if op == "radius":
        return [make("count_mismatch_fraction", count_mismatch(found, expected),
                     "below the neighbor cap both sides return complete "
                     "neighborhoods, so counts must match exactly")]
    if op == "voxelize-reduce":
        return [make("centroid_mismatch_fraction",
                     centroid_mismatch(found, expected),
                     "voxelize + reduce(mean) reproduces voxel_down_sample "
                     "bit for bit; centroids are quantized to identify voxels")]
    if op == "normals":
        return [make("max_normal_deviation", normal_deviation(found, expected),
                     "compared k-NN to k-NN and unsigned: Open3D's optional "
                     "radius changes its result and torchpcl's sign is "
                     "arbitrary", limit=0.01)]
    if op in OUTLIER_OPS:
        return [make("mask_mismatch_fraction", mask_mismatch(found, expected),
                     "torchpcl's k maps directly to Open3D's nb_neighbors / "
                     "nb_points, despite torchpcl counting the query point; "
                     "points within float32 noise of the threshold may fall "
                     "either way", limit=MASK_TOLERANCE),
                # Context, not a gate: agreement on a mask that keeps (or
                # drops) everything is trivially satisfied.
                make("removed_fraction", float(1.0 - found.mean()),
                     "fraction the filter actually removed; near 0 or 1 means "
                     "the comparison had little to disagree about",
                     bound="info")]
    descriptors, valid, limit_reached = found
    mismatch, worst_bin = descriptor_mismatch(descriptors, expected, valid)
    checks = [
        make("descriptor_mismatch_fraction", mismatch,
             f"rows disagreeing by more than {features.BIN_TOLERANCE} of 600 "
             "per row; a global max is dominated by single neighborhoods",
             limit=DESCRIPTOR_ROW_TOLERANCE),
        # Context: a single neighbor flipping bins costs about 100/k, so a
        # value near that is a rounding difference and a much larger one is not.
        make("max_bin_error", worst_bin,
             "worst single-bin difference, against 600 of mass per row",
             bound="info"),
    ]
    if limit_reached.mean() > 0:
        # Not a failure, but the comparison is then between two truncated
        # neighborhoods rather than two complete ones.
        checks.append(make("neighbor_limit_reached_fraction",
                           float(limit_reached.mean()),
                           "bounded search filled its neighbor list, so both "
                           "sides may summarize a truncated neighborhood",
                           bound="info"))
    return checks


def mapping_checks(prepared, op, device) -> list[Check]:
    """Confirm the outlier neighbor-count mapping actually discriminates.

    Exact agreement at `k` evidences the mapping only if the neighboring counts
    disagree. Without this, a comparison that had stopped distinguishing `k`
    from `k+-1` would still look perfect.
    """
    config = prepared.cloud.config
    found = _torchpcl_output(op, _run(prepared, op, "torchpcl", device), None)
    mismatches = []
    for offset in (-1, 1):
        shifted = config.outlier_k + offset
        if shifted < 1:
            continue
        neighbour = replace(prepared.cloud, config=replace(config, outlier_k=shifted))
        expected = _open3d_output(op, _run(
            replace(prepared, cloud=neighbour), op, "open3d", "cpu"))
        mismatches.append(mask_mismatch(found, expected))
    if not mismatches:
        return []
    return [Check(op, device, "offset_k_mismatch_fraction", min(mismatches),
                  "min", MAPPING_FLOOR, "open3d-cpu",
                  "k+-1 must disagree, or agreement at k would not evidence "
                  "the mapping; if this fails, suspect a workload where the "
                  "filter removes almost nothing or almost everything")]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=[*CHECKABLE_OPS, "all"], default="all")
    workload.add_arguments(parser)
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="cpu",
                        help="device for torchpcl; the reference is always CPU")
    parser.add_argument("--threads", type=positive_int, default=default_threads())
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.outlier_radius is None:
        args.outlier_radius = OUTLIER_RADIUS[args.input]
    workload.resolve_defaults(args, parser)
    return args


def _print_table(checks):
    print(f"\n{'op':22} {'device':7} {'metric':32} {'value':>11} "
          f"{'limit':>12}  status")
    for entry in checks:
        if entry.bound == "info":
            limit, status = "-", "info"
        else:
            limit = (("<=" if entry.bound == "max" else ">=")
                     + format(entry.limit, ".3g"))
            status = "ok" if entry.passed else "FAIL"
        print(f"{entry.op:22} {entry.device:7} {entry.metric:32} "
              f"{entry.value:11.3e} {limit:>12}  {status}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    o3d = require_open3d()
    import torch

    torch.set_num_threads(args.threads)
    from torchpcl import _C

    cuda_available = torch.cuda.is_available() and _C.has_cuda()
    try:
        devices, skipped = select_devices(args.device, cuda_available)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    ops = list(CHECKABLE_OPS) if args.op == "all" else [args.op]
    try:
        prepared = workload.prepare(args, ops)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(f"Checking torchpcl against Open3D {o3d.__version__} and brute force; "
          f"input: {args.input}; dtype: {args.dtype}; threads: {args.threads}")
    for line in prepared.describe():
        print(line)
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")

    checks = []
    for op in ops:
        reference = reference_output(prepared, op)
        for device in devices:
            checks.extend(check(prepared, op, device, reference))
            if op in OUTLIER_OPS:
                checks.extend(mapping_checks(prepared, op, device))
    _print_table(checks)

    failed = [entry for entry in checks if not entry.passed]
    print(f"\n{len(checks)} checks, {len(failed)} above tolerance")
    for entry in failed:
        print(f"  {entry.op}/{entry.device} {entry.metric}: {entry.reason}")

    report = {
        "environment": environment_metadata(devices, open3d_version=o3d.__version__),
        "config": {**prepared.config(), "threads": args.threads,
                   "devices": devices},
        "input": prepared.input_metadata(),
        "skipped": skipped,
        "checks": [entry.row() for entry in checks],
        "failed": len(failed),
    }
    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    raise SystemExit(1 if main()["failed"] else 0)
