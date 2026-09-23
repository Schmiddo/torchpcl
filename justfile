set positional-arguments

# The Python 3.12 environment holding Open3D and small-gicp. CC/CXX: uv's
# managed CPython names clang++, which the CUDA extension build rejects.
baseline_env := "CC=gcc CXX=g++ UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group baseline"

# Extra options such as `--device cuda` reach every run; --keep needs a clean tree.
# Measure every routine profile on this revision; --keep adds it to the history.
benchmark *args:
    uv run --group benchmark python -m benchmarks.bench "$@"

# Runs in the Python 3.12 baseline env; needed per library version, not per commit.
# Measure the third-party baselines of every routine profile.
benchmark-baselines *args:
    {{baseline_env}} python -m benchmarks.bench --baselines "$@"

# Suites: operations, sampling, registration, redwood. `just benchmark-run operations --op knn`
# Run one suite ad hoc, with any parameters; the report is not keepable.
benchmark-run *args:
    uv run --group benchmark python -m benchmarks.run "$@"

# Run one suite ad hoc in the baseline env, e.g. with `--backend open3d`.
benchmark-baseline *args:
    {{baseline_env}} python -m benchmarks.run "$@"

# Check torchpcl against Open3D and brute force. No timing; needs the Open3D env.
benchmark-accuracy *args:
    {{baseline_env}} python -m benchmarks.accuracy "$@"

# Diff two saved --json reports; needs neither Open3D nor the benchmark env.
benchmark-compare before after *args:
    uv run python benchmarks/compare.py "$@"

# Inspect or edit the kept history: add, list, drop REVISION, relabel OLD NEW.
benchmark-history *args:
    uv run python benchmarks/history.py "$@"

# Re-rendered on every reload, reachable from another computer; --out writes a file.
# Serve the kept history as a dashboard.
benchmark-dashboard *args:
    uv run python benchmarks/dashboard.py "$@"

download-data:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p data
    cd data
    printf '%s\n' \
        http://redwood-data.org/indoor/data/livingroom1-fragments-ply.zip \
        http://redwood-data.org/indoor/data/livingroom2-fragments-ply.zip \
        http://redwood-data.org/indoor/data/office1-fragments-ply.zip \
        http://redwood-data.org/indoor/data/office2-fragments-ply.zip \
        http://redwood-data.org/indoor/data/livingroom1-evaluation.zip \
        http://redwood-data.org/indoor/data/livingroom2-evaluation.zip \
        http://redwood-data.org/indoor/data/office1-evaluation.zip \
        http://redwood-data.org/indoor/data/office2-evaluation.zip \
        | uv run --with aria2 aria2c --input-file=- --max-concurrent-downloads=8 --continue=true
    for scene in livingroom1 livingroom2 office1 office2; do
        for kind in fragments-ply evaluation; do
            archive="$scene-$kind.zip"
            unzip -o "$archive" -d "${archive%.zip}"
            rm -- "$archive"
        done
    done
