"""
Train ML models on the INTERNET-CONNECTED staging machine.

Reads historical data (e.g., previously downloaded NVD + exploit data)
and trains ELP, ISA, ACC models, saving them as .joblib files for
transfer to the air-gapped host.

Usage:
    python -m staging.train_models --data-dir ./transfer/data --output-dir ./transfer/models
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_training_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix and label vectors from downloaded feed data.

    Returns:
        X       – feature matrix  (n_samples, n_features)
        y_elp   – binary exploit labels
        y_isa   – impact severity scores [0-10]
    """
    nvd_files = sorted(data_dir.glob("nvd_cves_*.json"))
    kev_files = sorted(data_dir.glob("cisa_kev_*.json"))

    # Collect CVE records
    cve_records = []
    for fp in nvd_files:
        with open(fp) as f:
            data = json.load(f)
        vulns = data.get("vulnerabilities", data.get("CVE_Items", []))
        cve_records.extend(vulns)

    # Build KEV set for labeling exploit status
    kev_ids: set[str] = set()
    for fp in kev_files:
        with open(fp) as f:
            data = json.load(f)
        for entry in data.get("vulnerabilities", []):
            kev_ids.add(entry.get("cveID", ""))

    if not cve_records:
        logger.warning("No CVE records found in %s", data_dir)
        return np.empty((0, 0)), np.empty(0), np.empty(0)

    features = []
    y_elp = []
    y_isa = []

    for item in cve_records:
        cve = item.get("cve", item)
        cve_id = cve.get("id", cve.get("CVE_data_meta", {}).get("ID", ""))

        # Extract CVSS v3 metrics where available
        metrics = cve.get("metrics", {})
        cvss_v3_list = metrics.get("cvssMetricV31", metrics.get("cvssMetricV30", []))
        if cvss_v3_list:
            cvss = cvss_v3_list[0].get("cvssData", {})
            base_score = cvss.get("baseScore", 0.0)
            exploitability = cvss.get("exploitabilityScore",
                                      cvss_v3_list[0].get("exploitabilityScore", 0.0))
            impact = cvss.get("impactScore",
                              cvss_v3_list[0].get("impactScore", 0.0))
        else:
            base_score = 0.0
            exploitability = 0.0
            impact = 0.0

        # Count references as proxy for exposure
        refs = cve.get("references", [])
        ref_count = len(refs) if isinstance(refs, list) else 0

        # Count affected configurations
        configs = cve.get("configurations", [])
        config_count = len(configs) if isinstance(configs, list) else 0

        # Simple feature vector
        row = [
            base_score,
            exploitability,
            impact,
            float(ref_count),
            float(config_count),
        ]
        features.append(row)
        y_elp.append(1.0 if cve_id in kev_ids else 0.0)
        y_isa.append(impact)

    X = np.array(features, dtype=np.float64)
    return X, np.array(y_elp), np.array(y_isa)


def train_and_save_models(
    X: np.ndarray,
    y_elp: np.ndarray,
    y_isa: np.ndarray,
    output_dir: Path,
) -> list[str]:
    """Train ELP, ISA, ACC models and save as .joblib."""
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier

    saved: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(X) < 10:
        logger.error("Insufficient training data (%d samples)", len(X))
        return saved

    # ── ELP (binary exploit likelihood) ──
    logger.info("Training ELP on %d samples ...", len(X))
    elp = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    feature_names = ["cvss_base_score", "exploitability", "impact", "ref_count", "config_count"]
    elp.fit(X, y_elp)
    elp_path = output_dir / "elp_1.0.0.joblib"
    joblib.dump({"model": elp, "feature_names": feature_names, "version": "1.0.0"}, elp_path)
    saved.append(str(elp_path))
    logger.info("ELP saved → %s", elp_path)

    # ── ISA (impact severity regressor) ──
    logger.info("Training ISA on %d samples ...", len(X))
    isa = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    isa.fit(X, y_isa)
    isa_path = output_dir / "isa_1.0.0.joblib"
    joblib.dump({"model": isa, "feature_names": feature_names, "version": "1.0.0"}, isa_path)
    saved.append(str(isa_path))
    logger.info("ISA saved → %s", isa_path)

    # ── ACC (asset criticality – synthetic labels from CVSS bands) ──
    logger.info("Training ACC on %d samples ...", len(X))
    tiers = ["low", "medium", "high", "critical"]
    y_acc_idx = np.digitize(X[:, 0], bins=[4.0, 7.0, 9.0])  # 0/1/2/3
    y_acc = np.array([tiers[i] for i in y_acc_idx])
    acc = RandomForestClassifier(
        n_estimators=150, max_depth=6, random_state=42, class_weight="balanced"
    )
    acc.fit(X, y_acc)
    acc_path = output_dir / "acc_1.0.0.joblib"
    joblib.dump({"model": acc, "feature_names": feature_names, "version": "1.0.0", "classes": tiers}, acc_path)
    saved.append(str(acc_path))
    logger.info("ACC saved → %s", acc_path)

    return saved


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train ML models on staging machine")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with downloaded feed data")
    parser.add_argument("--output-dir", type=Path, default=Path("./transfer/models"),
                        help="Output dir for trained .joblib models")
    args = parser.parse_args()

    X, y_elp, y_isa = build_training_data(args.data_dir)
    if X.size == 0:
        logger.error("No training data built – check data directory")
        return
    saved = train_and_save_models(X, y_elp, y_isa, args.output_dir)
    logger.info("Done. %d models saved to %s", len(saved), args.output_dir)


if __name__ == "__main__":
    main()
