import functools

import pytest
import torch

# `device` parametrizes pure-torch tests (transforms, estimation, voxel
# downsampling). Tests touching spatial search use
# `search_device` instead, which additionally skips when the compiled
# extension for that device is unavailable.
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=_DEVICES)
def device(request):
    return torch.device(request.param)


@functools.cache
def extension_skip_reason(device_type: str) -> str | None:
    """None if the search extension for device_type is usable, else a
    skip reason."""
    if device_type == "cuda":
        if not torch.cuda.is_available():
            return "CUDA not available"
        from torchpcl import search

        if search._cubql_cuda is None:
            return "torchpcl was installed without the CUDA extension"
    return None


@pytest.fixture(params=_DEVICES)
def search_device(request):
    """Device with a working torchpcl search extension, or skip."""
    if (reason := extension_skip_reason(request.param)) is not None:
        pytest.skip(reason)
    return torch.device(request.param)


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
