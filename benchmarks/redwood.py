"""Compare torchpcl and Open3D FGR on Redwood's ground-truth fragment pairs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.case import select_devices
from benchmarks.cli import (configure_threads, default_threads, environment_metadata,
                            positive_float, positive_int, require_open3d, write_json)
from benchmarks.registration_cases import RegistrationConfig, PreparedPair, prepare_cloud
from benchmarks.redwood_cases import SCENES, load_ground_truth, make_case, summarize
from benchmarks.runner import measure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--backend", choices=["torchpcl", "open3d"], default="torchpcl")
    parser.add_argument("--open3d-metrics", action="store_true",
                        help="also report Open3D fitness and inlier RMSE; needs Open3D")
    parser.add_argument("--device", choices=["all", "cpu", "cuda"], default="all",
                        help="torchpcl devices; Open3D FGR always uses CPU")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--voxel-size", type=positive_float, default=0.05)
    parser.add_argument("--fgr-distance", type=positive_float)
    parser.add_argument("--fgr-iterations", type=positive_int, default=64)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=positive_int, default=default_threads())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=positive_int,
                        help="evaluate only the first N GT pairs per scene (smoke test)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    if len(set(args.scenes)) != len(args.scenes):
        parser.error("--scenes must not contain duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_threads(args.threads)
    needs_open3d = args.backend == "open3d" or args.open3d_metrics
    o3d = require_open3d() if needs_open3d else None
    import torch

    torch.set_num_threads(args.threads)
    if args.backend == "torchpcl":
        from torchpcl import _C

        cuda_available = torch.cuda.is_available() and _C.has_cuda()
    else:
        # Open3D FGR is legacy CPU only, so CUDA is never available to it here.
        cuda_available = False
    try:
        devices, skipped = select_devices(args.device, cuda_available, backend=args.backend)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config = RegistrationConfig(voxel_size=args.voxel_size, dtype=args.dtype, seed=args.seed,
                                fgr_distance=args.fgr_distance,
                                fgr_iterations=args.fgr_iterations)
    # Validate every requested scene before starting expensive registration.
    truths = {}
    for scene in args.scenes:
        directory = args.data / f"{scene}-evaluation"
        if not directory.is_dir():
            raise SystemExit(f"Missing {directory}; run `just download-data` first")
        truths[scene] = load_ground_truth(directory)
        for index in range(truths[scene][0].fragments):
            path = args.data / f"{scene}-fragments-ply" / f"cloud_bin_{index}.ply"
            if not path.is_file():
                raise SystemExit(f"Missing {path}; run `just download-data` first")
    report = {
        "environment": environment_metadata(
            devices, open3d_version=o3d.__version__ if o3d is not None else None),
        "config": {**asdict(config), "data": str(args.data.resolve()), "scenes": args.scenes,
                   "devices": devices, "requested_device": args.device,
                   "threads": args.threads, "repeats": args.repeats, "warmup": args.warmup,
                   "max_pairs": args.max_pairs},
        "protocol": {"pairs": "non-consecutive ground-truth pairs only",
                     "error_squared_threshold": 0.04,
                     "timing": "registration only; shared Open3D preprocessing excluded",
                     "precision_evaluated": False},
        "skipped": skipped, "scenes": [],
    }
    print("Redwood FGR: GT pairs only; registration recall, not detection precision.")
    print(f"Backend: {args.backend}. Preprocessing is torchpcl's and excluded from "
          "timing; recall is scored against the dataset's ground truth.")
    if args.max_pairs:
        print(f"Subset run: at most {args.max_pairs} pairs per scene; recall is subset-only.")
    for skip in skipped:
        print(f"Skipped {skip['backend']}/{skip['device']}: {skip['reason']}")
    for scene, ground_truth in truths.items():
        @lru_cache(maxsize=8)
        def fragment(index):
            return prepare_cloud(args.data / f"{scene}-fragments-ply" / f"cloud_bin_{index}.ply",
                                 config)

        selected = ground_truth[:args.max_pairs]
        scene_report = {"scene": scene, "total_gt_pairs": len(ground_truth),
                        "evaluated_pairs": len(selected),
                        "subset": len(selected) < len(ground_truth), "results": []}
        report["scenes"].append(scene_report)
        for number, truth in enumerate(selected, 1):
            source, target = fragment(truth.source), fragment(truth.target)
            pair = PreparedPair(args.data / f"{scene}-fragments-ply", config, truth.transform,
                                (source[0], target[0]), (source[1], target[1]),
                                (source[2], target[2]))
            for device in devices:
                backend = args.backend
                case = make_case(pair, truth, backend, device,
                                 open3d_metrics=args.open3d_metrics)
                result = measure(case, repeats=args.repeats, warmup=args.warmup)
                result["pair"] = pair.metadata()
                scene_report["results"].append(result)
                successes = sum(s["registration_success"] for s in result["samples"])
                print(f"{scene} [{number}/{len(selected)}] {truth.source}->{truth.target} "
                      f"{backend}/{device}: {result['median']['time_ms']:.2f} ms, "
                      f"correct {successes}/{args.repeats}", flush=True)
                del case
        scene_report["summary"] = summarize(scene_report["results"])
        fragment.cache_clear()
        if args.json:
            write_json(report, args.json)
    report["summary"] = summarize([row for scene in report["scenes"] for row in scene["results"]])
    for summary in report["summary"]:
        print(f"{summary['backend']}/{summary['device']}: "
              f"registration recall {summary['mean_registration_recall']:.1%}, "
              f"{summary['pairs']} pairs, {summary['mean_pair_time_ms']:.2f} ms/pair")
    if args.json:
        write_json(report, args.json)
    return report


if __name__ == "__main__":
    main()
