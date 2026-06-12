import functools

import pytest
import torch

# Pure-torch modules (transforms, estimation, voxel downsampling, the
# brute-force metrics backend) work on both devices; spatial search and
# registration are CUDA-only.
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=_DEVICES)
def device(request):
    return torch.device(request.param)


@functools.cache
def cubql_skip_reason() -> str | None:
    """None if the CUDA extension is usable, else a skip reason.

    Cached so the JIT compile (or its failure) happens once per session.
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    try:
        from torchpcl.search import _load_extension

        _load_extension()
    except Exception as exc:  # noqa: BLE001 - any build failure means skip
        return f"torchpcl CUDA extension unavailable: {exc}"
    return None


@pytest.fixture
def cuda_device():
    """CUDA device with a working torchpcl extension, or skip."""
    if (reason := cubql_skip_reason()) is not None:
        pytest.skip(reason)
    return torch.device("cuda")


def random_cloud(n: int, device, scale: float = 1.0, seed: int | None = None) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    return (torch.rand((n, 3), generator=gen, dtype=torch.float64) * scale).to(device)


def random_rigid_transform(
    max_angle: float = 0.1, max_translation: float = 0.1, seed: int | None = None, device="cpu"
) -> torch.Tensor:
    from torchpcl.transforms import pose_to_matrix

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    pose = (torch.rand(6, generator=gen, dtype=torch.float64) * 2 - 1)
    pose[:3] *= max_angle
    pose[3:] *= max_translation
    return pose_to_matrix(pose).to(device)
