# torchpcl

`torchpcl` implements tensor-first point-cloud registration and processing for
PyTorch, in particular packed ragged batching and fast nearest-neighbor search on CPU
and CUDA.

## Installation

Binary wheels from tagged releases are available through the
[wheel index](https://schmiddo.github.io/torchpcl/wheels/). Select the index
matching the installed PyTorch and CUDA build; uv will then select the wheel
for the current Python and machine architecture.

For PyTorch 2.11.0 with CUDA 12.8:

```bash
uv pip install --no-deps torchpcl \
  --find-links https://schmiddo.github.io/torchpcl/wheels/torch2.11.0-cu128/
```

To build the native extension from source instead, build it against PyTorch in
the active environment with build isolation disabled:

```bash
pip install --no-build-isolation .
```

A CUDA build requires a CUDA toolkit compatible with the installed PyTorch.
Set `CUDA_HOME` to the toolkit location and use `TORCHPCL_WITH_CUDA=1` to
require CUDA support. To force a CPU-only build, use:

```bash
TORCHPCL_WITH_CUDA=0 pip install --no-build-isolation .
```

When installing from a Git checkout, initialize the cuBQL submodule first:

```bash
git submodule update --init third_party/cuBQL
```

## Documentation

- Read the [packed batching guide](batching.md) for point-cloud representation,
  devices, dtypes, and gradient behavior.
- Consult the [API reference](api.md) for the public Python interface.
- See the [repository README](https://github.com/Schmiddo/torchpcl#readme) for
  end-to-end examples and project details.
