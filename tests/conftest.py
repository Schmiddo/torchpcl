import pytest
import torch

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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
