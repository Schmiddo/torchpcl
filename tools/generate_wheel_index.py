"""Generate a uv-compatible flat index from GitHub release wheel assets."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
BUILD_VARIANT_PATTERN = re.compile(
    r"\+torch(?P<torch>[0-9]+(?:\.[0-9]+)*)\.(?P<cuda>cu[0-9]+)-"
    r"[^-]+-[^-]+-[^-]+\.whl$"
)


def fetch_releases(repository: str, token: str | None) -> list[dict[str, Any]]:
    """Fetch all releases in newest-first order from the GitHub REST API."""
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise ValueError("repository must have the form owner/name")

    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    releases: list[dict[str, Any]] = []
    page = 1

    while True:
        query = urllib.parse.urlencode({"per_page": 100, "page": page})
        url = f"{api_url}/repos/{repository}/releases?{query}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "torchpcl-wheel-index",
            "X-GitHub-Api-Version": "2026-03-10",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.load(response)
        if not isinstance(batch, list):
            raise RuntimeError("GitHub releases response was not a list")

        releases.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    return releases


def collect_wheels(
    releases: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Return unique wheel names, download URLs, and release tags."""
    wheels: list[tuple[str, str, str]] = []
    seen_names: set[str] = set()

    for release in releases:
        if release.get("draft"):
            continue
        tag = str(release.get("tag_name", ""))
        for asset in release.get("assets", []):
            name = str(asset.get("name", ""))
            url = str(asset.get("browser_download_url", ""))
            if not name.endswith(".whl") or not url or name in seen_names:
                continue

            digest = str(asset.get("digest", ""))
            if digest.startswith("sha256:"):
                url = f"{url}#sha256={digest.removeprefix('sha256:')}"

            wheels.append((name, url, tag))
            seen_names.add(name)

    return sorted(wheels)


def render_index(
    repository: str,
    wheels: list[tuple[str, str, str]],
    *,
    index_path: str = "wheels",
    title: str = "torchpcl wheel index",
    variant_indexes: list[tuple[str, str]] | None = None,
) -> str:
    """Render a valid HTML flat index understood by uv and pip."""
    if not wheels:
        raise RuntimeError(f"no wheel assets found in releases for {repository}")

    owner, project = repository.split("/", 1)
    index_url = f"https://{owner.lower()}.github.io/{project}/{index_path.strip('/')}/"
    links = "\n".join(
        "      <li>"
        f'<a href="{html.escape(url, quote=True)}" data-requires-python="&gt;=3.12">'
        f"{html.escape(name)}</a> "
        f"<small>({html.escape(tag)})</small></li>"
        for name, url, tag in wheels
    )
    command = f"uv pip install --no-deps torchpcl --find-links {index_url}"
    if variant_indexes:
        choices = "\n".join(
            f'        <li><a href="{html.escape(slug, quote=True)}/">'
            f"{html.escape(label)}</a></li>"
            for label, slug in variant_indexes
        )
        install_instructions = f"""<p>Select the index matching the installed
      PyTorch and CUDA build:</p>
      <ul>
{choices}
      </ul>"""
    else:
        install_instructions = f"<pre><code>{html.escape(command)}</code></pre>"

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
  </head>
  <body>
    <main>
      <h1>{html.escape(title)}</h1>
      <p>This is a flat package index backed by GitHub release assets.</p>
      {install_instructions}
      <p>Install a compatible PyTorch build before installing torchpcl.</p>
      <ul>
{links}
      </ul>
      <p><a href="../">Return to the documentation</a></p>
    </main>
  </body>
</html>
"""


def wheel_variant(name: str) -> tuple[str, str] | None:
    """Return the human-readable label and URL slug encoded in a local version."""
    match = BUILD_VARIANT_PATTERN.search(name)
    if match is None:
        return None
    torch_version = match.group("torch").removesuffix(".")
    cuda = match.group("cuda")
    return f"Torch {torch_version}, {cuda}", f"torch{torch_version}-{cuda}"


def group_variants(
    wheels: list[tuple[str, str, str]],
) -> dict[str, tuple[str, list[tuple[str, str, str]]]]:
    """Group wheels by the Torch/CUDA variant encoded in their local version."""
    variants: dict[str, tuple[str, list[tuple[str, str, str]]]] = {}
    for wheel in wheels:
        variant = wheel_variant(wheel[0])
        if variant is None:
            continue
        label, slug = variant
        variants.setdefault(slug, (label, []))[1].append(wheel)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, help="GitHub owner/name")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    releases = fetch_releases(args.repository, os.environ.get("GH_TOKEN"))
    wheels = collect_wheels(releases)
    variants = group_variants(wheels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_index(
            args.repository,
            wheels,
            variant_indexes=sorted(
                (label, slug) for slug, (label, _) in variants.items()
            ),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(wheels)} wheel links to {args.output}")

    for slug, (label, variant_wheels) in variants.items():
        variant_output = args.output.parent / slug / "index.html"
        variant_output.parent.mkdir(parents=True, exist_ok=True)
        variant_output.write_text(
            render_index(
                args.repository,
                variant_wheels,
                index_path=f"wheels/{slug}",
                title=f"torchpcl wheels for {label}",
            ),
            encoding="utf-8",
        )
        print(f"Wrote {len(variant_wheels)} wheel links to {variant_output}")


if __name__ == "__main__":
    main()
