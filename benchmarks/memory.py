"""Backend-neutral peak memory sampling for registration benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from threading import Event, Thread
from typing import Callable


_MIB = 1024**2


def process_rss_reader() -> Callable[[], int]:
    """Return a reader for this process's resident set size."""
    import psutil

    process = psutil.Process()
    return lambda: process.memory_info().rss


def cuda_process_memory_reader(device_index: int = 0) -> Callable[[], int]:
    """Return an NVML reader for this process on the selected CUDA device."""
    import pynvml
    import torch

    pynvml.nvmlInit()
    # CUDA device indices are logical and may be reordered by
    # CUDA_VISIBLE_DEVICES. Let PyTorch resolve the matching physical NVML
    # index; its device-properties UUID is an internal _CUuuid object rather
    # than the bytes/string accepted by pynvml.
    nvml_index = torch.cuda._get_nvml_device_index(device_index)
    handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
    pid = os.getpid()

    def read() -> int:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        used = [process.usedGpuMemory for process in processes if process.pid == pid]
        return max(used, default=0)

    return read


@dataclass(frozen=True)
class MemorySample:
    process_rss_bytes: int
    device_bytes: int | None


class PeakMemorySampler:
    """Poll process and optional per-process CUDA memory in a background thread."""

    def __init__(
        self,
        device_reader: Callable[[], int] | None = None,
        *,
        interval_seconds: float = 0.001,
        process_reader: Callable[[], int] | None = None,
    ) -> None:
        self._process_reader = process_rss_reader() if process_reader is None else process_reader
        self._device_reader = device_reader
        self._interval = interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self.baseline = self._read()
        self.peak = self.baseline

    def _read(self) -> MemorySample:
        return MemorySample(
            process_rss_bytes=self._process_reader(),
            device_bytes=None if self._device_reader is None else self._device_reader(),
        )

    def _update(self) -> None:
        sample = self._read()
        self.peak = MemorySample(
            process_rss_bytes=max(self.peak.process_rss_bytes, sample.process_rss_bytes),
            device_bytes=(
                None
                if sample.device_bytes is None
                else max(self.peak.device_bytes or 0, sample.device_bytes)
            ),
        )

    def _poll(self) -> None:
        while not self._stop.wait(self._interval):
            self._update()

    def __enter__(self) -> "PeakMemorySampler":
        self._thread = Thread(target=self._poll, name="benchmark-memory", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self._update()

    def report(self) -> dict[str, float | None]:
        device_baseline = self.baseline.device_bytes
        device_peak = self.peak.device_bytes
        return {
            "process_rss_baseline_mib": self.baseline.process_rss_bytes / _MIB,
            "process_rss_peak_mib": self.peak.process_rss_bytes / _MIB,
            "process_rss_delta_mib": (
                self.peak.process_rss_bytes - self.baseline.process_rss_bytes
            ) / _MIB,
            "device_baseline_mib": None if device_baseline is None else device_baseline / _MIB,
            "device_peak_mib": None if device_peak is None else device_peak / _MIB,
            "device_delta_mib": (
                None
                if device_baseline is None or device_peak is None
                else (device_peak - device_baseline) / _MIB
            ),
            "sampling_interval_ms": self._interval * 1000,
        }


def sample_peak_memory(case) -> dict[str, float | None]:
    """Measure one untimed, synchronized case execution."""
    case.reset()
    case.synchronize()
    with PeakMemorySampler(case.device_memory_reader) as sampler:
        result = case.run()
        case.synchronize()
        # Keep the result alive until the sampler records the final allocation state.
        _ = result
    return sampler.report()
