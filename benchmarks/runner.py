"""Optional wall-clock runner; cases can also be used by external harnesses."""

from statistics import median
from time import perf_counter

from .memory import sample_peak_memory


def measure(case, *, repeats=5, warmup=1):
    """Reset outside timing; synchronize on both boundaries; evaluate afterward."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    samples = []
    for repeat in range(warmup + repeats):
        case.reset()
        case.synchronize()
        start = perf_counter()
        result = case.run()
        case.synchronize()
        elapsed = 1000 * (perf_counter() - start)
        if repeat >= warmup:
            samples.append({"time_ms": elapsed, **case.evaluate(result)})
    keys = ("time_ms", "rotation_error_deg", "translation_error",
            "source_displacement_rmse", "fitness", "inlier_rmse")
    memory = sample_peak_memory(case)
    return {
        "method": case.method, "backend": case.backend, "device": case.device,
        "input_dtype": case.pair.config.dtype,
        "compute_dtype": (case.pair.config.dtype if case.backend == "torchpcl" else
                          "mixed" if case.method == "icp" and case.pair.config.dtype == "float32"
                          else "float64"),
        "implementation": "tensor" if case.backend == "open3d" and case.method == "icp" else
                          "legacy" if case.backend == "open3d" else "torchpcl",
        "reference_only": case.backend == "open3d" and case.method == "fgr",
        "transform_dtype": "float64" if case.backend == "open3d" else case.pair.config.dtype,
        "memory": memory,
        "median": {key: float(median(sample[key] for sample in samples)) for key in keys},
        "samples": samples,
    }
