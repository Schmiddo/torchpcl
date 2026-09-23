"""The benchmark suites: what each one measures, on which inputs.

A suite declares its workload arguments and yields cases; `run.py` owns
everything they have in common -- backend loading, device selection, timing,
the console table and the JSON report. A suite yields either a `Case` to
measure or a skip record, a dict naming what could not run and why.

Suites only prepare inputs here. The per-operation `*_cases` modules still
build each backend's call, so an external harness can reuse them without this
module or the runner.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from . import features_cases as features
from . import neighbors_cases as neighbors
from . import preprocess_cases as preprocess
from . import workload
from .case import Case
from .cli import positive_float, positive_int, require_open3d


def _grouped(case: Case, group: str) -> Case:
    # Set without holding a local reference, so the runner's `del` frees the
    # previous case's device memory before the next one is built.
    case.group = group
    return case


class Suite:
    name: str
    backends: tuple[str, ...]
    #: Arguments that choose which part of the work runs rather than what the
    #: work is. A profile leaves these open and pins everything else.
    selection: tuple[str, ...] = ()
    repeats = 5
    #: Whether cases' `evaluate` hooks run: only suites scored against ground
    #: truth, whose quality belongs next to their timing.
    evaluate = False
    #: Backends with no CUDA path in this suite, and why.
    cpu_only: dict[str, str] = {}

    def add_arguments(self, parser) -> None:
        raise NotImplementedError

    def resolve(self, args, parser) -> None:
        """Apply derived defaults and reject invalid combinations."""

    def columns(self, args) -> list[tuple[str, str, str]]:
        """Extra console columns: header, median metric name, format."""
        return []

    def cases(self, args, devices: list[str], report: dict) -> Iterator[Case | dict]:
        raise NotImplementedError

    def finish(self, report: dict) -> None:
        """Summarize measured results into the report, once all have run."""


class Operations(Suite):
    """Neighbor search, preprocessing and FPFH, one cloud shared by all."""

    name = "operations"
    backends = ("torchpcl", "open3d", "small-gicp")
    selection = ("op",)

    def add_arguments(self, parser):
        parser.add_argument("--op", choices=[*workload.ALL_OPS, "all"], default="all")
        workload.add_arguments(parser)

    def resolve(self, args, parser):
        workload.resolve_defaults(args, parser)

    def cases(self, args, devices, report):
        backend = args.backend
        ops = list(workload.ALL_OPS) if args.op == "all" else [args.op]
        supported = [op for op in ops if op in self._module(op).BACKEND_OPS[backend]]
        if args.op != "all" and not supported:
            # An explicit request for something a backend lacks is an error;
            # only a sweep may visibly skip it.
            raise SystemExit(f"{backend} does not implement {args.op}")
        for op in ops:
            if op not in supported:
                for device in devices:
                    yield {"backend": backend, "device": device, "op": op,
                           "reason": f"{backend} does not implement {op}"}
        try:
            prepared = workload.prepare(args, supported)
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
        report["input"] = {**prepared.input_metadata(),
                           "normals_source": prepared.normals_source}
        for line in prepared.describe():
            print(line)
        for op in supported:
            for device in devices:
                if op in neighbors.OPS:
                    yield neighbors.make_case(prepared.neighbors, op, backend, device,
                                              threads=args.threads)
                elif op in preprocess.OPS:
                    yield preprocess.make_case(prepared.cloud, op, backend, device,
                                               threads=args.threads)
                else:
                    yield features.make_case(prepared.features, op, backend, device)

    @staticmethod
    def _module(op):
        return (neighbors if op in neighbors.OPS else
                preprocess if op in preprocess.OPS else features)


class Sampling(Suite):
    """Farthest point sampling on a random cloud."""

    name = "sampling"
    backends = ("torchpcl", "open3d-tensor", "open3d-legacy")
    cpu_only = {"open3d-legacy": "Open3D legacy sampling supports CPU only"}

    def add_arguments(self, parser):
        parser.add_argument("--num-points", type=positive_int, default=200_000)
        parser.add_argument("--num-samples", type=positive_int, default=5_000)
        parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                            help="point precision; Open3D's legacy API always "
                                 "computes in float64")
        parser.add_argument("--seed", type=int, default=0)

    def resolve(self, args, parser):
        if args.backend == "open3d-legacy" and args.device == "cuda":
            parser.error(self.cpu_only["open3d-legacy"])
        if not 0 <= args.seed < 2**31:
            parser.error("--seed must be in [0, 2**31)")
        if args.num_samples >= args.num_points:
            parser.error("--num-samples must be less than --num-points")

    def cases(self, args, devices, report):
        from .sampling_cases import SamplingConfig, make_case, prepare_cloud

        prepared = prepare_cloud(SamplingConfig(
            num_points=args.num_points, num_samples=args.num_samples,
            dtype=args.dtype, seed=args.seed))
        report["input"] = prepared.metadata()
        if args.backend == "open3d-legacy":
            print("Open3D legacy sampling computes in float64.")
        for device in devices:
            yield make_case(prepared, args.backend, device)


def _registration_arguments(parser, *, icp: bool):
    parser.add_argument("--open3d-metrics", action="store_true",
                        help="also report Open3D fitness and inlier RMSE; needs Open3D")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="point input precision; Open3D FGR is legacy float64")
    parser.add_argument("--voxel-size", type=positive_float, default=0.05)
    if icp:
        parser.add_argument("--icp-distance", type=positive_float,
                            help="default: 5 * voxel size")
        parser.add_argument("--icp-iterations", type=positive_int, default=50)
    parser.add_argument("--fgr-distance", type=positive_float,
                        help="default: 0.5 * voxel size")
    parser.add_argument("--fgr-iterations", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=0)


def _check_seed(args, parser):
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")


class Registration(Suite):
    """ICP and FGR on pairs with a known rigid transform."""

    name = "registration"
    backends = ("torchpcl", "open3d", "small-gicp")
    selection = ("method", "open3d_metrics")
    evaluate = True

    def add_arguments(self, parser):
        parser.add_argument("--data", type=Path, nargs="+", default=[Path("data")],
                            help="pair directories containing source.ply, "
                                 "target.ply, T_target_source.txt")
        parser.add_argument("--objective", choices=["point-to-point", "point-to-plane"],
                            default="point-to-point", help="ICP residual; FGR ignores it")
        parser.add_argument("--method", choices=["all", "icp", "fgr"], default="all")
        _registration_arguments(parser, icp=True)

    def resolve(self, args, parser):
        _check_seed(args, parser)

    def columns(self, args):
        columns = [("rot deg", "rotation_error_deg", ".4f"),
                   ("trans", "translation_error", ".5f"),
                   ("GT RMSE", "source_displacement_rmse", ".5f")]
        return columns + ([("fitness", "fitness", ".4f")] if args.open3d_metrics else [])

    def cases(self, args, devices, report):
        from .registration_cases import RegistrationConfig, make_case, prepare_pair

        backend = args.backend
        if args.open3d_metrics:
            require_open3d()
        methods = ("icp", "fgr") if args.method == "all" else (args.method,)
        schedule = []
        for method in methods:
            for device in devices:
                if backend == "open3d" and method == "fgr" and device != "cpu":
                    yield {"backend": backend, "device": device, "method": method,
                           "reason": "Open3D FGR has no tensor or CUDA API"}
                elif backend == "small-gicp" and method == "fgr":
                    yield {"backend": backend, "device": device, "method": method,
                           "reason": "small-gicp has no global registration"}
                else:
                    schedule.append((method, device))

        config = RegistrationConfig(
            voxel_size=args.voxel_size, dtype=args.dtype, seed=args.seed,
            icp_distance=args.icp_distance, fgr_distance=args.fgr_distance,
            icp_iterations=args.icp_iterations, fgr_iterations=args.fgr_iterations)
        report["input"] = {"registration": asdict(config), "pairs": []}
        print("Registration only; preprocessing is torchpcl's and excluded from timing.")
        if args.objective == "point-to-plane":
            print("Point-to-plane uses the shared normals, except for small-gicp, "
                  "which has no normal setter and estimates its own.")
        if backend == "open3d" and "icp" in methods:
            print("Open3D tensor ICP stores points at the requested dtype/device; "
                  "its pose solver is CPU float64.")
        for directory in map(Path, args.data):
            pair = prepare_pair(directory, config,
                                with_normals=args.objective == "point-to-plane",
                                with_features="fgr" in methods)
            report["input"]["pairs"].append(pair.metadata())
            for method, device in schedule:
                yield _grouped(make_case(pair, method, backend, device,
                                         open3d_metrics=args.open3d_metrics,
                                         objective=args.objective,
                                         threads=args.threads), directory.name)


class Redwood(Suite):
    """FGR registration recall on Redwood's ground-truth fragment pairs."""

    name = "redwood"
    backends = ("torchpcl", "open3d")
    selection = ("open3d_metrics",)
    repeats = 1
    evaluate = True
    cpu_only = {"open3d": "Open3D FGR has no tensor or CUDA API"}

    def add_arguments(self, parser):
        from .redwood_cases import SCENES

        parser.add_argument("--data", type=Path, default=Path("data"))
        parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
        parser.add_argument("--max-pairs", type=positive_int,
                            help="evaluate only the first N GT pairs per scene "
                                 "(smoke test; recall is then subset-only)")
        _registration_arguments(parser, icp=False)

    def resolve(self, args, parser):
        _check_seed(args, parser)
        if len(set(args.scenes)) != len(args.scenes):
            parser.error("--scenes must not contain duplicates")

    def columns(self, args):
        return [("error²", "redwood_error_squared", ".4f")]

    def cases(self, args, devices, report):
        from .redwood_cases import load_ground_truth, make_case
        from .registration_cases import PreparedPair, RegistrationConfig, prepare_cloud

        if args.open3d_metrics:
            require_open3d()
        data = Path(args.data)
        config = RegistrationConfig(voxel_size=args.voxel_size, dtype=args.dtype,
                                    seed=args.seed, fgr_distance=args.fgr_distance,
                                    fgr_iterations=args.fgr_iterations)
        # Validate every requested scene before starting expensive registration.
        truths = {}
        for scene in args.scenes:
            directory = data / f"{scene}-evaluation"
            if not directory.is_dir():
                raise SystemExit(f"Missing {directory}; run `just download-data` first")
            truths[scene] = load_ground_truth(directory)
            for index in range(truths[scene][0].fragments):
                path = data / f"{scene}-fragments-ply" / f"cloud_bin_{index}.ply"
                if not path.is_file():
                    raise SystemExit(f"Missing {path}; run `just download-data` first")
        report["input"] = {
            "registration": asdict(config), "scenes": [],
            "protocol": {"pairs": "non-consecutive ground-truth pairs only",
                         "error_squared_threshold": 0.04,
                         "timing": "registration only; shared preprocessing excluded",
                         "precision_evaluated": False},
        }
        print("Redwood FGR: GT pairs only; registration recall, not detection precision.")
        if args.max_pairs:
            print(f"Subset run: at most {args.max_pairs} pairs per scene; "
                  "recall is subset-only.")
        for scene, ground_truth in truths.items():
            @lru_cache(maxsize=8)
            def fragment(index):
                return prepare_cloud(
                    data / f"{scene}-fragments-ply" / f"cloud_bin_{index}.ply", config)

            selected = ground_truth[:args.max_pairs]
            report["input"]["scenes"].append({
                "scene": scene, "total_gt_pairs": len(ground_truth),
                "evaluated_pairs": len(selected),
                "subset": len(selected) < len(ground_truth)})
            for truth in selected:
                source, target = fragment(truth.source), fragment(truth.target)
                pair = PreparedPair(data / f"{scene}-fragments-ply", config,
                                    truth.transform, (source[0], target[0]),
                                    (source[1], target[1]), (source[2], target[2]))
                for device in devices:
                    yield _grouped(make_case(pair, truth, args.backend, device,
                                             open3d_metrics=args.open3d_metrics),
                                   f"{scene}/{truth.target}-{truth.source}")
            fragment.cache_clear()

    def finish(self, report):
        from .redwood_cases import summarize

        results = report["results"]
        for scene in report["input"]["scenes"]:
            prefix = scene["scene"] + "/"
            scene["summary"] = summarize(
                [row for row in results if row["group"].startswith(prefix)])
        report["summary"] = summarize(results)
        for summary in report["summary"]:
            print(f"{summary['backend']}/{summary['device']}: "
                  f"registration recall {summary['mean_registration_recall']:.1%}, "
                  f"{summary['pairs']} pairs, {summary['mean_pair_time_ms']:.2f} ms/pair")


SUITES = {suite.name: suite for suite in
          (Operations(), Sampling(), Registration(), Redwood())}

__all__ = ["SUITES", "Suite"]
