import functools

import pytest
import torch

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@functools.cache
def cubql_skip_reason() -> str | None:
    """None if the cubql backend is usable, else a skip reason.

    Cached so the JIT compile (or its failure) happens once per session.
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    try:
        from torchpcl.search_cubql import _load_extension

        _load_extension()
    except Exception as exc:  # noqa: BLE001 - any build failure means skip
        return f"cubql backend unavailable: {exc}"
    return None


@pytest.fixture(params=_DEVICES)
def device(request):
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
