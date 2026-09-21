set positional-arguments

# Measure ICP/FGR in the project environment (torchpcl by default).
benchmark-registration *args:
    uv run --group benchmark python benchmarks/registration.py "$@"

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

# Run the self-contained sweep on the GPU only.
benchmark-gpu *args:
    #!/usr/bin/env bash
    set -euo pipefail
    # An explicit --device cuda errors out on a build without CUDA, where the
    # default --device all would quietly measure the CPU alone and leave you
    # comparing a GPU run against a CPU one.
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p runs
    just benchmark-sampling --device cuda --json "runs/$stamp-sampling-cuda.json" "$@"
    just benchmark-operations --device cuda \
        --json "runs/$stamp-operations-random-cuda.json" "$@"
    echo "Reports: runs/$stamp-*-cuda.json"

# Measure every third-party baseline on the self-contained workloads.
benchmark-baselines *args:
    #!/usr/bin/env bash
    set -uo pipefail
    # Each backend selects the devices it supports, so this covers the GPU
    # wherever Open3D has one and records small-gicp's CPU-only skip. One leg
    # failing must not lose the legs that already measured, so they are all
    # attempted and the failures reported at the end.
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p runs
    extra=("$@")
    failed=()
    # script, backend, report name. Sampling is the only workload where Open3D's
    # legacy and tensor APIs are distinct backends rather than one.
    for leg in "operations open3d operations-random-open3d" \
               "operations small-gicp operations-random-small-gicp" \
               "sampling open3d-tensor sampling-open3d-tensor" \
               "sampling open3d-legacy sampling-open3d-legacy"; do
        read -r script backend name <<<"$leg"
        echo
        echo "=== $script --backend $backend ==="
        just benchmark-baseline "$script" --backend "$backend" \
            --json "runs/$stamp-$name.json" "${extra[@]}" || failed+=("$script/$backend")
    done
    echo
    echo "Reports: runs/$stamp-*.json"
    if [ ${#failed[@]} -gt 0 ]; then
        echo "Did not measure: ${failed[*]}" >&2
        exit 1
    fi

# Check torchpcl against Open3D and brute force. No timing; needs the Open3D env.
benchmark-accuracy *args:
    just benchmark-baseline accuracy "$@"

# Run any benchmark script in the Python 3.12 Open3D environment, for the two
# things Open3D is still for: `--backend open3d` baselines and accuracy checks.
benchmark-baseline script *args:
    shift; CC=gcc CXX=g++ UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group baseline python benchmarks/{{script}}.py "$@"

# Diff two saved --json reports; needs neither Open3D nor the benchmark env.
benchmark-compare before after *args:
    uv run python benchmarks/compare.py "$@"

# Promote a saved report into the kept history (runs/ is scratch; this is not).
benchmark-keep *args:
    uv run python benchmarks/history.py add "$@"

# Re-render runs/dashboard.html from the kept history.
benchmark-dashboard *args:
    uv run python benchmarks/dashboard.py "$@"

# Evaluate FGR recall on Redwood ground-truth pairs. Ground truth comes from
# the dataset, so this needs no Open3D.
benchmark-redwood *args:
    uv run --group benchmark python benchmarks/redwood.py "$@"

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
