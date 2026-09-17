set positional-arguments

# Run the registration benchmark in a separate Python 3.12 environment.
benchmark-registration *args:
    CC=gcc CXX=g++ UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group baseline python benchmarks/registration.py "$@"

# Measure sampling in the project environment (torchpcl by default).
benchmark-sampling *args:
    uv run --group benchmark python benchmarks/sampling.py "$@"

# Measure per-operation kernels (torchpcl by default).
benchmark-operations *args:
    uv run --group benchmark python benchmarks/operations.py "$@"

# Run self-contained random benchmarks, saving timestamped reports into runs/.
benchmark *args:
    #!/usr/bin/env bash
    set -euo pipefail
    # One stamp for the whole sweep, so a run's reports group together and
    # sort chronologically; UTC to match the `timestamp` inside each report.
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p runs
    just benchmark-sampling --json "runs/$stamp-sampling.json" "$@"
    just benchmark-operations --json "runs/$stamp-operations-random.json" "$@"
    echo "Reports: runs/$stamp-*.json"

# Check torchpcl against Open3D and brute force. No timing; needs the Open3D env.
benchmark-accuracy *args:
    UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group baseline python benchmarks/accuracy.py "$@"

# Diff two saved --json reports; needs neither Open3D nor the benchmark env.
benchmark-compare before after *args:
    uv run python benchmarks/compare.py "$@"

# Evaluate FGR on Redwood ground-truth fragment pairs against Open3D.
benchmark-redwood *args:
    UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group baseline python benchmarks/redwood.py "$@"

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
