set positional-arguments

# Run the registration benchmark in a separate Python 3.12 environment.
benchmark *args:
    UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group benchmark python benchmarks/registration.py "$@"

# Run the sampling benchmark in a separate Python 3.12 environment.
benchmark-sampling *args:
    UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group benchmark python benchmarks/sampling.py "$@"

# Evaluate FGR on Redwood ground-truth fragment pairs against Open3D.
benchmark-redwood *args:
    UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group benchmark python benchmarks/redwood.py "$@"

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
