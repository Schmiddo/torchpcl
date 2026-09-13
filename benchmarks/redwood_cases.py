"""Redwood ground-truth loading and FGR cases sharing registration inputs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .registration_cases import PreparedPair, make_case as registration_case

SCENES = ("livingroom1", "livingroom2", "office1", "office2")


@dataclass(frozen=True)
class GroundTruth:
    target: int
    source: int
    fragments: int
    transform: np.ndarray
    information: np.ndarray


def _read_blocks(path, size):
    lines = Path(path).read_text().splitlines()
    lines = [line for line in lines if line.strip()]
    if not lines or len(lines) % (size + 1):
        raise ValueError(f"{path}: incomplete or empty matrix records")
    records = {}
    for start in range(0, len(lines), size + 1):
        header = tuple(map(int, lines[start].split()))
        if len(header) != 3 or not 0 <= header[0] < header[1] < header[2]:
            raise ValueError(f"{path}: invalid fragment header {header}")
        matrix = np.array([list(map(float, row.split()))
                           for row in lines[start + 1:start + size + 1]])
        if matrix.shape != (size, size) or not np.isfinite(matrix).all():
            raise ValueError(f"{path}: expected finite {size}x{size} matrix")
        if header in records:
            raise ValueError(f"{path}: duplicate pair {header}")
        records[header] = matrix
    if len({header[2] for header in records}) != 1:
        raise ValueError(f"{path}: inconsistent fragment counts")
    return records


def load_ground_truth(directory):
    """Match records by header, excluding consecutive pairs as Redwood does."""
    directory = Path(directory)
    transforms = _read_blocks(directory / "gt.log", 4)
    information = _read_blocks(directory / "gt.info", 6)
    if transforms.keys() != information.keys():
        raise ValueError(f"{directory}: gt.log and gt.info pair headers differ")
    pairs = []
    for (i, j, count), transform in transforms.items():
        info = information[i, j, count]
        rotation = transform[:3, :3]
        if not (np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5)
                and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                and np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError(f"{directory}: non-rigid transform for {(i, j)}")
        if info[0, 0] <= 0 or not np.allclose(info, info.T):
            raise ValueError(f"{directory}: invalid information matrix for {(i, j)}")
        if j > i + 1:
            pairs.append(GroundTruth(i, j, count, transform, info))
    if not pairs:
        raise ValueError(f"{directory}: no non-consecutive ground-truth pairs")
    return pairs


def transformation_error(estimate, expected, information):
    """Squared Redwood error: [translation, quaternion xyz], normalized by info[0,0].

    Matches mrComputeTransformationError in ElasticReconstruction's Matlab
    toolbox, using a stable quaternion conversion also at 180 degrees.
    """
    relative = np.linalg.solve(expected, estimate)
    r = relative[:3, :3]
    # Largest-component conversion avoids division by a near-zero quaternion w.
    candidates = np.array([1 + np.trace(r), 1 + 2*r[0, 0] - np.trace(r),
                           1 + 2*r[1, 1] - np.trace(r),
                           1 + 2*r[2, 2] - np.trace(r)])
    k = int(np.argmax(candidates))
    q = np.zeros(4)
    q[k] = np.sqrt(max(0., candidates[k])) / 2
    denominator = 4 * q[k]
    if k == 0:
        q[1:] = [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]
        q[1:] /= denominator
    else:
        a = k - 1
        b, c = (a + 1) % 3, (a + 2) % 3
        q[0] = (r[c, b] - r[b, c]) / denominator
        q[b + 1] = (r[b, a] + r[a, b]) / denominator
        q[c + 1] = (r[c, a] + r[a, c]) / denominator
    if q[0] < 0:
        q = -q
    error = np.r_[relative[:3, 3], q[1:]]
    return float(max(0., error @ information @ error / information[0, 0]))


def make_case(pair: PreparedPair, truth: GroundTruth, backend, device="cpu"):
    case = registration_case(pair, "fgr", backend, device)
    evaluate_registration = case.evaluate

    def evaluate(result):
        metrics = evaluate_registration(result)
        error = transformation_error(np.asarray(metrics["transform"]),
                                     truth.transform, truth.information)
        return {**metrics, "redwood_error_squared": error,
                "registration_success": error <= 0.04}

    case.name = "redwood-fgr"
    case.evaluate = evaluate
    case.metadata.update(target_fragment=truth.target, source_fragment=truth.source)
    return case


def summarize(results):
    """Aggregate each repeat separately; do not turn median poses into successes."""
    groups = {}
    for result in results:
        groups.setdefault((result["backend"], result["device"]), []).append(result)
    summaries = []
    for (backend, device), rows in groups.items():
        repeats = len(rows[0]["samples"])
        successes = [sum(row["samples"][r]["registration_success"] for row in rows)
                     for r in range(repeats)]
        summaries.append({"backend": backend, "device": device, "pairs": len(rows),
                          "successful_pairs_per_repeat": successes,
                          "registration_recall_per_repeat": [n / len(rows) for n in successes],
                          "mean_registration_recall": float(np.mean(successes) / len(rows)),
                          "mean_pair_time_ms": float(np.mean([row["median"]["time_ms"]
                                                              for row in rows]))})
    return summaries
