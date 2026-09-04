# torchpcl

torchpcl provides tensor-first point-cloud registration and processing for
PyTorch, with packed ragged batching and fast nearest-neighbor search on CPU
and CUDA.

## Installation

The native extension is compiled against PyTorch in the active environment,
so build isolation must be disabled:

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
