# torchpcl

`torchpcl` implements common point-cloud processing routines for PyTorch, in particular fast nearest-neighbor search and registration, on CPU and CUDA.

The API is not final yet and may change.

## Installation

Binary wheels are available through the [wheel index](https://schmiddo.github.io/torchpcl/wheels/).
One wheel serves every PyTorch >= 2.10 and Python >= 3.12.
Select the index matching your PyTorch build: `cu130` (CUDA 13), `cu126` (CUDA 12.6+), or `cpu`.
A CUDA wheel does not load with a CPU-only PyTorch.

```bash
uv pip install --no-deps torchpcl \
  --find-links https://schmiddo.github.io/torchpcl/wheels/cu130/
```

With [`uv`](https://docs.astral.sh/uv/), the index can also be added to your project's `pyproject.toml`:

```toml
[project]
dependencies = ["torchpcl"]

[tool.uv]
find-links = ["https://schmiddo.github.io/torchpcl/wheels/cu130/"]
```

Earlier releases keep their per-PyTorch indexes, such as `wheels/torch2.14.0-cu130/`.

To build from source, clone with the cuBQL submodule:

```bash
git clone --recurse-submodules https://github.com/Schmiddo/torchpcl.git
cd torchpcl
uv sync
```

Set `CUDA_HOME` to your CUDA toolkit; `TORCHPCL_WITH_CUDA=1` requires a CUDA
build and `TORCHPCL_WITH_CUDA=0` forces CPU-only.

## Documentation

- Read the [packed batching guide](batching.md) for the point-cloud data model,
  devices, dtypes, and gradient behavior.
- Consult the [API reference](api.md) for the available operations and public
  Python interface.
- See the [repository README](https://github.com/Schmiddo/torchpcl#readme) for
  end-to-end examples and development instructions.
