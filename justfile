set positional-arguments

# Run the registration benchmark in a separate Python 3.12 environment.
benchmark *args:
    UV_PROJECT_ENVIRONMENT=.venv-benchmark uv run --python 3.12 --group benchmark python benchmarks/registration.py "$@"
