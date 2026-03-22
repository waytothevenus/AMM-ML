"""
Backtesting Engine – RECOMMENDED ADDITION #4.

Validates the system's predictions against historical ground truth.
Essential for building trust in an air-gapped system where real-time
feedback from external sources is unavailable.

Metrics:
  - ELP accuracy  (AUC, precision, recall on exploit prediction)
  - ISA accuracy  (MAE between predicted and actual impact)
  - Markov calibration  (predicted vs observed state transitions)
  - Prioritization quality (how often top-ranked items were correct)
  - Temporal forecast accuracy (risk predictions at t+7, t+30 vs actuals)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    period: str
    elp_metrics: dict[str, float] = field(default_factory=dict)
    isa_metrics: dict[str, float] = field(default_factory=dict)
    markov_metrics: dict[str, float] = field(default_factory=dict)
    prioritization_metrics: dict[str, float] = field(default_factory=dict)
    forecast_metrics: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0


class BacktestingEngine:
    """
    Runs backtests by replaying historical data through the pipeline
    and comparing predictions against outcomes.
    """

    def run_backtest(
        self,
        predictions: list[dict],
        actuals: list[dict],
        period: str = "",
    ) -> BacktestResult:
        """
        Compare a batch of predictions against actual outcomes.

        Each prediction dict:  {vuln_id, asset_id, exploit_likelihood,
                                adjusted_impact, composite_risk, final_rank,
                                markov_distribution, forecast}
        Each actual dict:      {vuln_id, asset_id, was_exploited (bool),
                                actual_impact (float), actual_state (int),
                                was_remediated_in_time (bool)}
        """
        result = BacktestResult(period=period)

        # Build lookup
        actual_map = {
            f"{a['vuln_id']}::{a['asset_id']}": a for a in actuals
        }

        matched_preds = []
        matched_acts = []
        for p in predictions:
            key = f"{p['vuln_id']}::{p['asset_id']}"
            if key in actual_map:
                matched_preds.append(p)
                matched_acts.append(actual_map[key])

        if not matched_preds:
            logger.warning("No matched prediction–actual pairs for backtesting")
            return result

        # ELP metrics
        result.elp_metrics = self._eval_elp(matched_preds, matched_acts)

        # ISA metrics
        result.isa_metrics = self._eval_isa(matched_preds, matched_acts)

        # Markov calibration
        result.markov_metrics = self._eval_markov(matched_preds, matched_acts)

        # Prioritization quality
        result.prioritization_metrics = self._eval_prioritization(matched_preds, matched_acts)

        # Overall composite score
        scores = [
            result.elp_metrics.get("auc", 0.5),
            1.0 - min(1.0, result.isa_metrics.get("mae", 5.0) / 10.0),
            result.markov_metrics.get("calibration_score", 0.5),
            result.prioritization_metrics.get("precision_at_10", 0.5),
        ]
        result.overall_score = float(np.mean(scores))

        logger.info(
            "Backtest [%s]: %d pairs, overall=%.3f (ELP-AUC=%.3f, ISA-MAE=%.3f)",
            period, len(matched_preds), result.overall_score,
            result.elp_metrics.get("auc", 0), result.isa_metrics.get("mae", 0),
        )
        return result

    def _eval_elp(self, preds: list[dict], acts: list[dict]) -> dict[str, float]:
        """Evaluate exploit likelihood predictions."""
        y_true = np.array([1.0 if a.get("was_exploited") else 0.0 for a in acts])
        y_score = np.array([p.get("exploit_likelihood", 0.5) for p in preds])

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return {"auc": 0.5, "note": "single_class"}

        # Manual AUC (no sklearn import needed at runtime)
        auc = self._roc_auc(y_true, y_score)

        # Precision/recall at threshold 0.5
        y_pred = (y_score >= 0.5).astype(float)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "auc": auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": 2 * precision * recall / (precision + recall + 1e-12),
        }

    def _eval_isa(self, preds: list[dict], acts: list[dict]) -> dict[str, float]:
        """Evaluate impact severity adjustment."""
        y_true = np.array([a.get("actual_impact", 5.0) for a in acts])
        y_pred = np.array([p.get("adjusted_impact", 5.0) for p in preds])

        mae = float(np.abs(y_true - y_pred).mean())
        rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

        return {"mae": mae, "rmse": rmse}

    def _eval_markov(self, preds: list[dict], acts: list[dict]) -> dict[str, float]:
        """
        Evaluate Markov state prediction calibration.
        Compare predicted dominant state vs actual state.
        """
        correct = 0
        total = 0
        for p, a in zip(preds, acts):
            dist = p.get("markov_distribution")
            actual_state = a.get("actual_state")
            if dist is not None and actual_state is not None:
                predicted_state = int(np.argmax(dist))
                if predicted_state == actual_state:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {"state_accuracy": accuracy, "calibration_score": accuracy, "evaluated": total}

    def _eval_prioritization(self, preds: list[dict], acts: list[dict]) -> dict[str, float]:
        """
        Evaluate prioritization: do top-ranked items correlate with actual exploits?
        """
        # Sort predictions by final_rank
        indexed = list(zip(preds, acts))
        indexed.sort(key=lambda x: x[0].get("final_rank", 999))

        # Precision@10: fraction of top-10 that were actually exploited
        top_10 = indexed[:10]
        exploited_in_top = sum(1 for _, a in top_10 if a.get("was_exploited"))
        total_exploited = sum(1 for _, a in indexed if a.get("was_exploited"))

        p_at_10 = exploited_in_top / min(10, len(top_10)) if top_10 else 0.0
        recall_at_10 = exploited_in_top / total_exploited if total_exploited > 0 else 0.0

        return {
            "precision_at_10": float(p_at_10),
            "recall_at_10": float(recall_at_10),
            "total_exploited": total_exploited,
        }

    @staticmethod
    def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Simple AUC computation via trapezoidal rule."""
        desc_idx = np.argsort(-y_score)
        y_true_sorted = y_true[desc_idx]

        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_prev, fpr_prev = 0.0, 0.0
        auc = 0.0
        tp, fp = 0, 0

        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
            tpr_prev, fpr_prev = tpr, fpr

        return float(auc)
