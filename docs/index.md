# torchpcl

`torchpcl` implements common point-cloud processing routines for PyTorch, in particular fast nearest-neighbor search and registration.
It strives to provide an easy-to-use, yet performant developer experience.

The API is not final yet and might change at any time.

## Installation

We recommend [`uv`](https://docs.astral.sh/uv/) for installation.
Binary wheels from tagged releases are available through the [wheel index](https://schmiddo.github.io/torchpcl/wheels/).
Select the index matching your installed PyTorch and CUDA build; uv will then select the wheel for the current Python and machine architecture.

Add the selected wheel index to your project's `pyproject.toml`:

```toml
[project]
dependencies = ["torchpcl"]

[tool.uv]
find-links = [
  "https://schmiddo.github.io/torchpcl/wheels/torch2.14.0-cu130/",
]
```

For example, for PyTorch 2.14.0 with CUDA 13.0:

```bash
uv pip install --no-deps torchpcl \
  --find-links https://schmiddo.github.io/torchpcl/wheels/torch2.14.0-cu130/
```

Alternatively, install the newest version from source. The cuBQL headers are
the only Git submodule required by the build:

```bash
git clone --recurse-submodules https://github.com/Schmiddo/torchpcl.git
cd torchpcl
uv sync
```

Ensure that `CUDA_HOME` points to your CUDA toolkit location. Use
`TORCHPCL_WITH_CUDA=1` to force a GPU-enabled build and
`TORCHPCL_WITH_CUDA=0` for a CPU-only build.

## Documentation

- Read the [packed batching guide](batching.md) for the point-cloud data model,
  devices, dtypes, and gradient behavior.
- Consult the [API reference](api.md) for the available operations and public
  Python interface.
- See the [repository README](https://github.com/Schmiddo/torchpcl#readme) for
  end-to-end examples and development instructions.
