"""
Download Script – runs on the INTERNET-CONNECTED staging machine.

Downloads all required data feeds and packages them into a transfer
bundle that can be moved to the air-gapped host via USB/file share.

Usage (on staging machine):
    python -m staging.download_feeds --output-dir /path/to/usb/transfer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# NOTE: These URLs are the official public endpoints for vulnerability data.
# They are used ONLY on the internet-connected staging machine.
FEED_SOURCES = {
    "nvd": {
        "url": "https://services.nvd.nist.gov/rest/json/cves/2.0",
        "description": "NVD CVE feed (API 2.0)",
        "filename": "nvd_cves_{date}.json",
    },
    "cisa_kev": {
        "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "alt_url": "https://cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "description": "CISA Known Exploited Vulnerabilities",
        "filename": "cisa_kev_{date}.json",
    },
    "exploitdb": {
        "description": "Exploit-DB CSV (manually download from exploit-db.com/download)",
        "filename": "exploitdb_files_{date}.csv",
        "manual": True,
    },
}


def create_transfer_bundle(output_dir: Path) -> Path:
    """
    Create a structured transfer directory ready for USB copy.

    Structure:
        transfer_YYYYMMDD/
        ├── manifest.json
        ├── data/
        │   ├── nvd_cves_20240601.json
        │   ├── cisa_kev_20240601.json
        │   └── ...
        ├── models/
        │   ├── elp_1.0.0.joblib
        │   ├── isa_1.0.0.joblib
        │   └── acc_1.0.0.joblib
        └── checksums.sha256
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    bundle_dir = output_dir / f"transfer_{date_str}"
    data_dir = bundle_dir / "data"
    models_dir = bundle_dir / "models"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "staging.download_feeds",
        "feeds": {},
        "models": [],
    }

    # Download feeds
    for feed_name, feed_info in FEED_SOURCES.items():
        if feed_info.get("manual"):
            logger.info(
                "MANUAL: %s – %s", feed_name, feed_info["description"]
            )
            manifest["feeds"][feed_name] = {"status": "manual_required"}
            continue

        filename = feed_info["filename"].format(date=date_str)
        dest = data_dir / filename

        urls = [feed_info["url"]]
        if feed_info.get("alt_url"):
            urls.append(feed_info["alt_url"])

        downloaded = False
        last_exc = None
        for url in urls:
            try:
                _download_file(url, dest)
                manifest["feeds"][feed_name] = {
                    "status": "downloaded",
                    "filename": filename,
                    "size_bytes": dest.stat().st_size,
                }
                downloaded = True
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("URL failed (%s), trying next: %s", url, exc)

        if not downloaded:
            logger.error("Failed to download %s: %s", feed_name, last_exc)
            manifest["feeds"][feed_name] = {"status": "failed", "error": str(last_exc)}

    # Write manifest
    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate checksums
    _generate_checksums(bundle_dir)

    logger.info("Transfer bundle created: %s", bundle_dir)
    return bundle_dir


def _download_file(url: str, dest: Path) -> None:
    """Download a URL to a local file."""
    import urllib.request
    import ssl

    ctx = ssl.create_default_context()
    logger.info("Downloading %s → %s", url, dest.name)

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; VulnRiskAssessment/1.0)",
        "Accept": "application/json, */*",
    })
    with urllib.request.urlopen(req, context=ctx, timeout=120) as response:
        with open(dest, "wb") as f:
            shutil.copyfileobj(response, f)

    logger.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)


def _generate_checksums(bundle_dir: Path) -> None:
    """Generate SHA-256 checksums for all files in the bundle."""
    import hashlib

    checksum_lines = []
    for fp in sorted(bundle_dir.rglob("*")):
        if fp.is_file() and fp.name != "checksums.sha256":
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            rel = fp.relative_to(bundle_dir)
            checksum_lines.append(f"{h.hexdigest()}  {rel}")

    checksum_path = bundle_dir / "checksums.sha256"
    checksum_path.write_text("\n".join(checksum_lines) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download feeds for air-gap transfer")
    parser.add_argument("--output-dir", type=Path, default=Path("./transfer"),
                        help="Output directory for transfer bundle")
    args = parser.parse_args()
    create_transfer_bundle(args.output_dir)


if __name__ == "__main__":
    main()
