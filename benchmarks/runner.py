"""Optional wall-clock runner; cases can also be used by external harnesses."""

from statistics import median
from time import perf_counter

from .memory import sample_peak_memory


def measure(case, *, repeats=5, warmup=1):
    """Reset outside timing; synchronize on both boundaries; evaluate afterward.

    ``case.evaluate``'s numeric fields (bools excluded) are medianed across
    repetitions; other fields (e.g. a serialized transform) are kept only in
    ``samples``. ``case.metadata`` is reported once, unmeasured.

    ``time_ms_min``/``time_ms_max`` bound the measured runtimes. A median
    alone cannot show whether two backends' runtimes actually differ, which
    matters most on CUDA, where clock and thermal drift move repetitions far
    more than on CPU.
    """
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
    numeric_keys = [
        key for key, value in samples[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    memory = sample_peak_memory(case)
    times = [sample["time_ms"] for sample in samples]
    return {
        "name": case.name, "backend": case.backend, "device": case.device,
        **case.metadata,
        "memory": memory,
        "time_ms_min": min(times),
        "time_ms_max": max(times),
        "median": {key: float(median(sample[key] for sample in samples)) for key in numeric_keys},
        "samples": samples,
    }
