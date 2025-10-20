from __future__ import annotations

from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from models import RiskScore, SurveillancePeriod
from datetime import datetime


class RiskScoringModel:
    """
    Minimal risk scoring facade to satisfy service imports and provide
    a consistent output structure used by premium calculations and dashboards.
    """

    def __init__(self, db: Session):
        self.db = db

    def calculate_comprehensive_risk_score(self, institution_id: str, period_id: Optional[str]) -> Dict[str, Any]:
        """
        Returns a dict with key 'risk_metrics' containing 'composite_risk_score'.
        Attempts to read an existing RiskScore for the period, otherwise falls back
        to a neutral mid value.
        """
        composite = 0.5  # neutral default
        try:
            query = self.db.query(RiskScore).join(SurveillancePeriod)
            if period_id:
                score = query.filter(RiskScore.period_id == period_id).first()
            else:
                score = (
                    query.filter(SurveillancePeriod.institution_id == institution_id)
                    .order_by(RiskScore.calculated_at.desc())
                    .first()
                )
            if score and score.composite_score is not None:
                composite = float(score.composite_score)
        except Exception:
            # fall back to default if DB not ready
            composite = 0.5

        return {
            "model": "heuristic_fallback",
            "evaluated_at": datetime.utcnow().isoformat(),
            "risk_metrics": {
                "composite_risk_score": composite,
            },
        }
