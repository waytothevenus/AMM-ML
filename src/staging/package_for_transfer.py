"""
Package-for-Transfer – creates a self-contained transfer archive.

Collects downloaded data, trained models, and Python wheel dependencies
into a single directory (or .zip) ready for USB/file-share transfer to
the air-gapped host.

Usage:
    python -m staging.package_for_transfer \
        --data-dir  ./transfer/data   \
        --model-dir ./transfer/models \
        --output    ./transfer_bundle.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_wheels(dest_dir: Path, requirements: list[str] | None = None) -> int:
    """
    Download Python wheels for offline installation.

    Runs `pip download` to collect .whl files into dest_dir.
    Should be run on the staging machine WITH internet access.
    """
    import subprocess

    dest_dir.mkdir(parents=True, exist_ok=True)

    if requirements is None:
        requirements = [
            "numpy", "scipy", "scikit-learn", "xgboost",
            "networkx", "duckdb", "pydantic>=2", "fastapi",
            "uvicorn[standard]", "strawberry-graphql",
            "joblib", "pandas", "pyyaml",
        ]

    cmd = [
        "pip", "download",
        "--dest", str(dest_dir),
        "--only-binary", ":all:",
    ] + requirements

    logger.info("Downloading wheels: %s", " ".join(requirements))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("pip download returned %d:\n%s", result.returncode, result.stderr)
    else:
        logger.info("Wheels downloaded to %s", dest_dir)

    return len(list(dest_dir.glob("*.whl")))


def build_package(
    data_dir: Path,
    model_dir: Path,
    output_path: Path,
    include_wheels: bool = True,
) -> Path:
    """
    Build the transfer package.

    Layout inside the archive:
        transfer/
        ├── manifest.json
        ├── checksums.sha256
        ├── data/          # NVD, KEV, ExploitDB, etc.
        ├── models/        # .joblib model files
        └── wheels/        # Python wheel dependencies (optional)
    """
    staging = output_path.parent / "_staging_temp"
    if staging.exists():
        shutil.rmtree(staging)

    pkg = staging / "transfer"
    pkg.mkdir(parents=True)

    manifest: dict = {
        "created_at": datetime.utcnow().isoformat(),
        "contents": {},
    }

    # ── Copy data files ──
    if data_dir.exists():
        dest = pkg / "data"
        shutil.copytree(data_dir, dest)
        manifest["contents"]["data"] = [
            f.name for f in sorted(dest.iterdir()) if f.is_file()
        ]
        logger.info("Copied %d data files", len(manifest["contents"]["data"]))

    # ── Copy models ──
    if model_dir.exists():
        dest = pkg / "models"
        shutil.copytree(model_dir, dest)
        manifest["contents"]["models"] = [
            f.name for f in sorted(dest.iterdir()) if f.is_file()
        ]
        logger.info("Copied %d model files", len(manifest["contents"]["models"]))

    # ── Collect wheels ──
    if include_wheels:
        wheels_dir = pkg / "wheels"
        n = collect_wheels(wheels_dir)
        manifest["contents"]["wheels"] = n
        logger.info("Collected %d wheel files", n)

    # ── Write manifest ──
    with open(pkg / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Write checksums ──
    lines = []
    for fp in sorted(pkg.rglob("*")):
        if fp.is_file() and fp.name != "checksums.sha256":
            rel = fp.relative_to(pkg)
            lines.append(f"{compute_sha256(fp)}  {rel}")
    (pkg / "checksums.sha256").write_text("\n".join(lines) + "\n")

    # ── Create .zip ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(pkg.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(staging))

    shutil.rmtree(staging)
    logger.info("Transfer package created: %s (%.1f MB)",
                output_path, output_path.stat().st_size / 1_048_576)
    return output_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Package data for air-gap transfer")
    parser.add_argument("--data-dir", type=Path, default=Path("./transfer/data"))
    parser.add_argument("--model-dir", type=Path, default=Path("./transfer/models"))
    parser.add_argument("--output", type=Path, default=Path("./transfer_bundle.zip"))
    parser.add_argument("--no-wheels", action="store_true",
                        help="Skip downloading Python wheels")
    args = parser.parse_args()

    build_package(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_path=args.output,
        include_wheels=not args.no_wheels,
    )


if __name__ == "__main__":
    main()
