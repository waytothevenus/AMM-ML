"""
Risk Decay Calculator – models how risk decreases after mitigation events.

Once a vulnerability enters the Mitigated or Remediated state, the
residual risk doesn't drop to zero instantly.  This module computes
post-remediation risk decay curves.
"""

from __future__ import annotations

import math
from datetime import datetime


class RiskDecayCalculator:
    """
    Exponential risk decay after mitigation/remediation.

    risk(t) = residual_base × exp(-λ × days_since_action)

    Parameters
    ----------
    half_life_mitigated : float
        Half-life (days) for residual risk after mitigation (workaround applied).
    half_life_remediated : float
        Half-life (days) for residual risk after full remediation (patch applied).
    """

    def __init__(
        self,
        half_life_mitigated: float = 14.0,
        half_life_remediated: float = 3.0,
        residual_base_mitigated: float = 0.3,
        residual_base_remediated: float = 0.05,
    ) -> None:
        self._hl_mit = half_life_mitigated
        self._hl_rem = half_life_remediated
        self._base_mit = residual_base_mitigated
        self._base_rem = residual_base_remediated
        # Decay constant: λ = ln(2) / half_life
        self._lambda_mit = math.log(2) / half_life_mitigated if half_life_mitigated > 0 else 0
        self._lambda_rem = math.log(2) / half_life_remediated if half_life_remediated > 0 else 0

    def residual_risk_mitigated(self, days_since_mitigation: float) -> float:
        """Risk remaining after a mitigation action (workaround)."""
        return self._base_mit * math.exp(-self._lambda_mit * max(0.0, days_since_mitigation))

    def residual_risk_remediated(self, days_since_remediation: float) -> float:
        """Risk remaining after a full remediation (patch)."""
        return self._base_rem * math.exp(-self._lambda_rem * max(0.0, days_since_remediation))

    def compute_decay(
        self,
        action_type: str,
        action_date: datetime | str,
        reference_time: datetime | None = None,
    ) -> float:
        """
        Compute current residual risk given an action type and date.

        action_type: "mitigated" or "remediated"
        """
        now = reference_time or datetime.utcnow()
        if isinstance(action_date, str):
            action_date = datetime.fromisoformat(action_date)

        days = (now - action_date).total_seconds() / 86400.0
        days = max(0.0, days)

        if action_type == "remediated":
            return self.residual_risk_remediated(days)
        else:
            return self.residual_risk_mitigated(days)

    def time_to_negligible(
        self,
        action_type: str,
        threshold: float = 0.01,
    ) -> float:
        """
        Days until residual risk drops below `threshold`.
        """
        if action_type == "remediated":
            base, lam = self._base_rem, self._lambda_rem
        else:
            base, lam = self._base_mit, self._lambda_mit

        if lam <= 0 or base <= threshold:
            return 0.0
        # threshold = base * exp(-λt) → t = -ln(threshold/base) / λ
        return -math.log(threshold / base) / lam
