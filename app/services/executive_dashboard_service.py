import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from models import SurveillancePeriod, DepositAnalysis, SurveillanceExposureCalculation as ExposureCalculation, CAMELSRating, Institution, RiskScore, EarlyWarningSignal
from app.services.deposit_analysis_service import DepositAnalysisService
from app.services.exposure_calculation_service import ExposureCalculationService
from camels_calculations import CAMELSCalculations
from app.services.risk_scoring import RiskScoringModel

class ExecutiveDashboardService:
    def __init__(self, db: Session):
        self.db = db
        self.deposit_service = DepositAnalysisService(db)
        self.exposure_service = ExposureCalculationService(db)
        self.camels_calculations = CAMELSCalculations(db)
        self.risk_scoring = RiskScoringModel(db)

    async def generate_executive_dashboard(self, institution_id: str,
                                          period_type: str = "QUARTERLY",
                                          num_periods: int = 4) -> Dict[str, Any]:
        """Generate comprehensive executive dashboard data"""
        
        institution = self.db.query(Institution).filter(Institution.id == institution_id).first()
        if not institution:
            return {"error": "Institution not found"}

        dashboard_data = {
            "institution_name": institution.name,
            "overview_stats": await self._get_overview_stats(institution_id),
            "deposit_trends": await self._get_deposit_trends(institution_id, period_type, num_periods),
            "exposure_summary": await self._get_exposure_summary(institution_id, period_type, num_periods),
            "camels_summary": await self._get_camels_summary(institution_id, period_type, num_periods),
            "risk_profile": await self._get_risk_profile(institution_id, period_type, num_periods),
            "last_updated": datetime.utcnow().isoformat()
        }

        return dashboard_data

    async def _get_overview_stats(self, institution_id: str) -> Dict[str, Any]:
        """Get high-level overview statistics"""
        
        latest_deposit_analysis = self.db.query(DepositAnalysis).join(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id
        ).order_by(SurveillancePeriod.analysis_date.desc()).first()

        latest_camels = self.db.query(CAMELSRating).join(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id
        ).order_by(SurveillancePeriod.analysis_date.desc()).first()

        latest_risk_score = self.db.query(RiskScore).join(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id
        ).order_by(RiskScore.calculated_at.desc()).first()

        return {
            "total_deposits": float(latest_deposit_analysis.total_deposits) if latest_deposit_analysis else 0.0,
            "total_accounts": latest_deposit_analysis.total_accounts if latest_deposit_analysis else 0,
            "latest_camels_composite": float(latest_camels.composite_rating) if latest_camels else None,
            "latest_camels_risk_grade": latest_camels.risk_grade if latest_camels else "N/A",
            "latest_risk_score": float(latest_risk_score.composite_score) if latest_risk_score else None,
            "latest_risk_grade": latest_risk_score.risk_grade if latest_risk_score else "N/A"
        }

    async def _get_deposit_trends(self, institution_id: str, period_type: str, num_periods: int) -> Dict[str, Any]:
        """Get historical deposit trends"""
        
        periods = self.db.query(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id,
            SurveillancePeriod.period_type == period_type
        ).order_by(SurveillancePeriod.analysis_date.desc()).limit(num_periods).all()
        
        periods.reverse() # Oldest first

        total_deposits_trend = []
        individual_deposits_trend = []
        corporate_deposits_trend = []
        period_labels = []

        for period in periods:
            deposit_analyses = self.db.query(DepositAnalysis).filter(DepositAnalysis.period_id == period.id).all()
            
            total_deposits = sum(float(da.total_deposits) for da in deposit_analyses)
            individual_deposits = sum(float(da.total_deposits) for da in deposit_analyses if da.deposit_type == "INDIVIDUAL")
            corporate_deposits = sum(float(da.total_deposits) for da in deposit_analyses if da.deposit_type == "CORPORATE")

            total_deposits_trend.append(total_deposits)
            individual_deposits_trend.append(individual_deposits)
            corporate_deposits_trend.append(corporate_deposits)
            period_labels.append(period.period_end.strftime("%Y-%m"))
        
        return {
            "labels": period_labels,
            "total_deposits": total_deposits_trend,
            "individual_deposits": individual_deposits_trend,
            "corporate_deposits": corporate_deposits_trend
        }

    async def _get_exposure_summary(self, institution_id: str, period_type: str, num_periods: int) -> Dict[str, Any]:
        """Get summary of exposure calculations"""
        
        periods = self.db.query(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id,
            SurveillancePeriod.period_type == period_type
        ).order_by(SurveillancePeriod.analysis_date.desc()).limit(num_periods).all()
        
        periods.reverse()

        total_insured_trend = []
        overall_exposure_trend = []
        period_labels = []

        for period in periods:
            exposure_calcs = self.db.query(ExposureCalculation).filter(ExposureCalculation.period_id == period.id).all()
            
            total_insured = sum(float(ec.insured_amount) for ec in exposure_calcs)
            total_deposits = sum(float(ec.total_deposits) for ec in exposure_calcs)
            overall_exposure = (total_insured / total_deposits) * 100 if total_deposits > 0 else 0

            total_insured_trend.append(total_insured)
            overall_exposure_trend.append(overall_exposure)
            period_labels.append(period.period_end.strftime("%Y-%m"))
        
        return {
            "labels": period_labels,
            "total_insured": total_insured_trend,
            "overall_exposure_percentage": overall_exposure_trend
        }

    async def _get_camels_summary(self, institution_id: str, period_type: str, num_periods: int) -> Dict[str, Any]:
        """Get summary of CAMELS ratings"""
        
        periods = self.db.query(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id,
            SurveillancePeriod.period_type == period_type
        ).order_by(SurveillancePeriod.analysis_date.desc()).limit(num_periods).all()
        
        periods.reverse()

        composite_rating_trend = []
        capital_trend = []
        asset_trend = []
        management_trend = []
        earnings_trend = []
        liquidity_trend = []
        sensitivity_trend = []
        period_labels = []

        for period in periods:
            camels_rating = self.db.query(CAMELSRating).filter(CAMELSRating.period_id == period.id).first()
            if camels_rating:
                composite_rating_trend.append(float(camels_rating.composite_rating))
                capital_trend.append(float(camels_rating.capital_adequacy))
                asset_trend.append(float(camels_rating.asset_quality))
                management_trend.append(float(camels_rating.management_quality))
                earnings_trend.append(float(camels_rating.earnings))
                liquidity_trend.append(float(camels_rating.liquidity))
                sensitivity_trend.append(float(camels_rating.sensitivity))
                period_labels.append(period.period_end.strftime("%Y-%m"))
        
        return {
            "labels": period_labels,
            "composite_rating": composite_rating_trend,
            "capital_adequacy": capital_trend,
            "asset_quality": asset_trend,
            "management_quality": management_trend,
            "earnings": earnings_trend,
            "liquidity": liquidity_trend,
            "sensitivity": sensitivity_trend
        }

    async def _get_risk_profile(self, institution_id: str, period_type: str, num_periods: int) -> Dict[str, Any]:
        """Get summary of risk scores and early warnings"""
        
        latest_risk_score = self.db.query(RiskScore).join(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id
        ).order_by(RiskScore.calculated_at.desc()).first()

        latest_early_warnings = self.db.query(EarlyWarningSignal).filter(
            EarlyWarningSignal.institution_id == institution_id
        ).order_by(EarlyWarningSignal.triggered_at.desc()).limit(5).all()

        return {
            "latest_overall_risk_score": float(latest_risk_score.composite_score) if latest_risk_score else None,
            "latest_risk_grade": latest_risk_score.risk_grade if latest_risk_score else "N/A",
            "early_warning_signals": [
                {
                    "signal_type": ews.signal_type,
                    "severity": ews.severity,
                    "description": ews.description,
                    "triggered_at": ews.triggered_at.isoformat()
                } for ews in latest_early_warnings
            ]
        }