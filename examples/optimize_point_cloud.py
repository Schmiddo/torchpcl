#!/usr/bin/env python3
"""Deform a point cloud to match a target with Chamfer distance.

The source point coordinates are the optimization parameters.  When no source
file is given, the source starts as random points within the target's bounding
box.  Inputs may be PLY files or NPZ files containing a ``points`` array.  The
script writes the initial cloud, regular optimization snapshots, and the target
cloud as PLY files.  Only point positions are carried into the output; input
colors and normals are ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement, PlyParseError

import torchpcl as tp


def xyz_array(values: np.ndarray, name: str, path: Path) -> np.ndarray:
    """Convert an array to contiguous, finite float32 xyz vectors."""
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name!r} in {path} must be a numeric array") from error
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name!r} in {path} must have shape (N, 3)")
    if len(result) == 0:
        raise ValueError(f"{name!r} in {path} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name!r} in {path} contains non-finite values")
    return np.ascontiguousarray(result)


def read_ply(path: Path) -> np.ndarray:
    """Read x/y/z vertices from a PLY file with plyfile."""
    try:
        ply_data = PlyData.read(path)
    except (OSError, PlyParseError) as error:
        raise ValueError(f"could not read PLY file {path}") from error

    try:
        vertices = ply_data["vertex"]
    except KeyError as error:
        raise ValueError(f"{path} does not contain a vertex element") from error

    property_names = {prop.name for prop in vertices.properties}
    if not {"x", "y", "z"}.issubset(property_names):
        raise ValueError(f"{path} vertices must have x, y, and z properties")
    return xyz_array(
        np.column_stack([vertices[axis] for axis in ("x", "y", "z")]),
        "points",
        path,
    )


def read_npz(path: Path) -> np.ndarray:
    """Read points and validate optional normals from an NPZ archive."""
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read NPZ file {path}") from error

    with archive:
        if "points" not in archive:
            raise ValueError(f"{path} does not contain a 'points' array")
        points = xyz_array(archive["points"], "points", path)
        if "normals" in archive:
            normals = xyz_array(archive["normals"], "normals", path)
            if normals.shape != points.shape:
                raise ValueError(
                    f"'normals' in {path} must have the same shape as 'points'"
                )
    return points


def read_point_cloud(path: Path) -> np.ndarray:
    """Read point positions from a supported point-cloud file."""
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return read_ply(path)
    if suffix == ".npz":
        return read_npz(path)
    raise ValueError(f"unsupported point-cloud file extension {suffix!r}: {path}")


def write_ply(path: Path, points: torch.Tensor) -> None:
    """Write a tensor of point positions as a binary PLY file with plyfile."""
    positions = points.detach().to(device="cpu", dtype=torch.float32).numpy()
    vertices = np.empty(
        len(positions), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    for column, axis in enumerate(("x", "y", "z")):
        vertices[axis] = positions[:, column]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def random_source(target: torch.Tensor, seed: int) -> torch.Tensor:
    """Sample one source point per target point inside its bounding box."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unit_points = torch.rand(
        (len(target), 3), generator=generator, dtype=target.dtype, device="cpu"
    ).to(target.device)
    lower = target.amin(dim=0)
    upper = target.amax(dim=0)
    return lower + unit_points * (upper - lower)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        from torchpcl import _C

        requested = "cuda" if torch.cuda.is_available() and _C.has_cuda() else "cpu"

    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("torchpcl point-cloud search supports CPU and CUDA devices")
    if device.type == "cuda":
        from torchpcl import _C

        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA device")
        if not _C.has_cuda():
            raise ValueError("CUDA was requested, but torchpcl was built without CUDA")
    return device


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="PLY or NPZ cloud to match")
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        help="optional PLY or NPZ cloud to optimize; defaults to a random cloud",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("point_cloud_optimization"),
        help="snapshot directory (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=1_000,
        help="number of optimizer steps (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=1e-2,
        help="optimizer learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "sgd"),
        default="adam",
        help="point-coordinate optimizer (default: %(default)s)",
    )
    parser.add_argument(
        "--save-every",
        type=positive_int,
        default=100,
        help="save a PLY snapshot every N steps (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, cuda:N, or auto (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random source seed when no source file is given (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = torch.from_numpy(read_point_cloud(args.target)).to(device)
    if args.source is None:
        initial_source = random_source(target, args.seed)
        source_origin = f"random(seed={args.seed})"
    else:
        initial_source = torch.from_numpy(read_point_cloud(args.source)).to(device)
        source_origin = str(args.source)
    source = torch.nn.Parameter(initial_source)
    optimizer_type = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_type([source], lr=args.learning_rate)

    write_ply(args.output_dir / "target.ply", target)
    write_ply(args.output_dir / "source_000000.ply", source)
    with torch.no_grad():
        initial_loss = tp.chamfer_distance(source, target, squared=True).item()
    print(
        f"device={device} optimizer={args.optimizer} source={source_origin} "
        f"source_points={len(source)} target_points={len(target)}\n"
        f"step=0 loss={initial_loss:.8g} file=source_000000.ply"
    )

    for step in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = tp.chamfer_distance(source, target, squared=True)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Chamfer loss became non-finite at step {step}")
        backward_loss = loss * len(source) if args.optimizer == "sgd" else loss
        backward_loss.backward()
        if source.grad is None or not bool(torch.isfinite(source.grad).all()):
            raise RuntimeError(f"source gradient became non-finite at step {step}")
        optimizer.step()

        if step % args.save_every == 0 or step == args.iterations:
            snapshot_name = f"source_{step:06d}.ply"
            write_ply(args.output_dir / snapshot_name, source)
            with torch.no_grad():
                snapshot_loss = tp.chamfer_distance(
                    source, target, squared=True
                ).item()
            print(f"step={step} loss={snapshot_loss:.8g} file={snapshot_name}")


if __name__ == "__main__":
    main()
