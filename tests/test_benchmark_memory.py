import sys
from types import SimpleNamespace

import pytest

from benchmarks import memory


def test_cuda_reader_maps_logical_device_to_nvml_index(monkeypatch):
    calls = []
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlDeviceGetHandleByIndex=lambda index: calls.append(("handle", index)) or 7,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [],
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(_get_nvml_device_index=lambda index: index + 4)
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    reader = memory.cuda_process_memory_reader(2)

    assert calls == ["init", ("handle", 6)]
    assert reader() == 0


def test_peak_memory_report_tracks_absolute_and_incremental_usage():
    rss = iter([100, 140, 130])
    device = iter([200, 260, 250])

    sampler = memory.PeakMemorySampler(
        lambda: next(device), interval_seconds=0.01, process_reader=lambda: next(rss)
    )
    sampler._update()
    sampler._update()
    report = sampler.report()

    mib = 1024**2
    assert report["process_rss_baseline_mib"] == pytest.approx(100 / mib)
    assert report["process_rss_peak_mib"] == pytest.approx(140 / mib)
    assert report["process_rss_delta_mib"] == pytest.approx(40 / mib)
    assert report["device_baseline_mib"] == pytest.approx(200 / mib)
    assert report["device_peak_mib"] == pytest.approx(260 / mib)
    assert report["device_delta_mib"] == pytest.approx(60 / mib)


def test_cpu_memory_report_has_no_device_values():
    sampler = memory.PeakMemorySampler(process_reader=lambda: 1024**2)
    sampler._update()
    report = sampler.report()
    assert report["process_rss_peak_mib"] == 1
    assert report["device_baseline_mib"] is None
    assert report["device_peak_mib"] is None
    assert report["device_delta_mib"] is None
