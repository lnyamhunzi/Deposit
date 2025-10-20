import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
import uuid
from app.models.surveillance import ExposureCalculation, SurveillancePeriod, DepositType
from app.services.deposit_analysis_service import DepositAnalysisService
from app.models.returns import ReturnUpload, ReturnPeriod, Institution

class ExposureCalculationService:
    def __init__(self, db: Session):
        self.db = db
        self.deposit_service = DepositAnalysisService(db)

    async def calculate_exposure(self, institution_id: str, period_id: str,
                                cover_level: Decimal = Decimal('100000')) -> Dict[str, Any]:
        """Calculate exposure for an institution based on deposit data"""
        
        period = self.db.query(SurveillancePeriod).filter(SurveillancePeriod.id == period_id).first()
        if not period:
            return {"error": "Surveillance Period not found"}

        # Get deposit data for the period
        deposit_data_df = await self.deposit_service._extract_deposit_data(
            institution_id, period.period_start, period.period_end
        )

        if deposit_data_df is None or deposit_data_df.empty:
            return {"error": "No deposit data available for exposure calculation"}

        results = []
        total_insured_amount = Decimal('0')
        total_uninsured_amount = Decimal('0')
        total_deposits = Decimal('0')

        for deposit_type in DepositType:
            type_data = deposit_data_df[deposit_data_df['account_type'].str.contains(deposit_type.value, case=False, na=False)]
            
            if not type_data.empty:
                type_total_deposits = Decimal(str(type_data['balance'].sum()))
                type_insured_amount = Decimal(str(type_data['balance'].apply(lambda x: min(x, cover_level)).sum()))
                type_uninsured_amount = type_total_deposits - type_insured_amount

                total_deposits += type_total_deposits
                total_insured_amount += type_insured_amount
                total_uninsured_amount += type_uninsured_amount

                # Create ExposureCalculation record
                exposure_record = ExposureCalculation(
                    id=str(uuid.uuid4()),
                    period_id=period.id,
                    deposit_type=deposit_type,
                    total_deposits=type_total_deposits,
                    insured_amount=type_insured_amount,
                    uninsured_amount=type_uninsured_amount,
                    cover_level=cover_level,
                    calculation_date=datetime.utcnow()
                )
                self.db.add(exposure_record)
                results.append(exposure_record)
        
        self.db.commit()

        # Calculate overall exposure percentage
        overall_exposure_percentage = (total_insured_amount / total_deposits) * 100 if total_deposits > 0 else Decimal('0')

        return {
            "institution_id": institution_id,
            "period_id": period_id,
            "cover_level": float(cover_level),
            "total_deposits_analyzed": float(total_deposits),
            "total_insured_amount": float(total_insured_amount),
            "total_uninsured_amount": float(total_uninsured_amount),
            "overall_exposure_percentage": float(overall_exposure_percentage),
            "exposure_by_type": [
                {
                    "deposit_type": res.deposit_type.value,
                    "total_deposits": float(res.total_deposits),
                    "insured_amount": float(res.insured_amount),
                    "uninsured_amount": float(res.uninsured_amount)
                } for res in results
            ],
            "calculation_date": datetime.utcnow().isoformat()
        }

    async def get_exposure_calculations_for_period(self, period_id: str) -> List[ExposureCalculation]:
        """Get all exposure calculations for a given surveillance period"""
        return self.db.query(ExposureCalculation).filter(ExposureCalculation.period_id == period_id).all()

    async def get_latest_exposure_calculation(self, institution_id: str) -> Optional[ExposureCalculation]:
        """Get the latest exposure calculation for an institution"""
        return self.db.query(ExposureCalculation).join(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id
        ).order_by(ExposureCalculation.calculation_date.desc()).first()

    async def _calculate_concentration_risk(self, analysis: Dict) -> Decimal:
        """Calculate concentration risk for deposit type"""
        
        # Risk factors: large accounts, high average balance, low diversification
        large_accounts_ratio = Decimal('0')
        if analysis.get('account_size_breakdown'):
            large_count = analysis['account_size_breakdown'].get('LARGE', 0)
            large_accounts_ratio = Decimal(large_count) / Decimal(analysis['total_accounts'])
        
        # Higher average balance indicates concentration
        avg_balance_risk = min(Decimal('1'), analysis['average_balance'] / Decimal('50000'))
        
        # Currency concentration
        currency_concentration = Decimal('0')
        if analysis.get('currency_breakdown'):
            max_currency_share = max(analysis['currency_breakdown'].values()) / float(analysis['total_deposits'])
            currency_concentration = Decimal(str(max_currency_share))
        
        concentration_risk = (large_accounts_ratio * Decimal('0.4') + 
                            avg_balance_risk * Decimal('0.3') + 
                            currency_concentration * Decimal('0.3'))
        
        return concentration_risk
    
    async def _calculate_volatility_risk(self, deposit_type: DepositType) -> Decimal:
        """Calculate volatility risk based on historical data"""
        # This would query historical volatility
        # For now, use type-based risk factors
        volatility_factors = {
            DepositType.INDIVIDUAL: Decimal('0.2'),
            DepositType.CORPORATE: Decimal('0.6'),
            DepositType.GOVERNMENT: Decimal('0.1'),
            DepositType.JOINT: Decimal('0.3'),
            DepositType.TRUST: Decimal('0.4')
        }
        
        return volatility_factors.get(deposit_type, Decimal('0.5'))
    
    async def _assess_exposure_risk(self, calculations: List[ExposureCalculation]) -> Dict[str, Any]:
        """Assess overall exposure risk"""
        
        total_insured = sum(float(calc.insured_amount) for calc in calculations)
        max_concentration = max(float(calc.concentration_risk) for calc in calculations)
        avg_volatility = np.mean([float(calc.volatility_risk) for calc in calculations])
        
        # Risk scoring
        concentration_score = max_concentration * 100
        volatility_score = avg_volatility * 100
        overall_risk_score = (concentration_score * 0.6 + volatility_score * 0.4)
        
        risk_level = "LOW"
        if overall_risk_score > 70:
            risk_level = "HIGH"
        elif overall_risk_score > 40:
            risk_level = "MEDIUM"
        
        return {
            "overall_risk_score": overall_risk_score,
            "risk_level": risk_level,
            "concentration_score": concentration_score,
            "volatility_score": volatility_score,
            "total_exposure": total_insured,
            "key_risk_factors": await self._identify_key_risk_factors(calculations)
        }
    
    async def _identify_key_risk_factors(self, calculations: List[ExposureCalculation]) -> List[Dict[str, Any]]:
        """Identify key risk factors in exposure"""
        
        risk_factors = []
        
        for calc in calculations:
            if float(calc.concentration_risk) > 0.7:
                risk_factors.append({
                    "type": "CONCENTRATION",
                    "deposit_type": calc.deposit_type.value,
                    "severity": "HIGH",
                    "description": f"High concentration in {calc.deposit_type.value} deposits",
                    "metric": float(calc.concentration_risk)
                })
            
            if float(calc.volatility_risk) > 0.6:
                risk_factors.append({
                    "type": "VOLATILITY", 
                    "deposit_type": calc.deposit_type.value,
                    "severity": "MEDIUM",
                    "description": f"High volatility in {calc.deposit_type.value} deposits",
                    "metric": float(calc.volatility_risk)
                })
        
        return sorted(risk_factors, key=lambda x: x["severity"], reverse=True)
    
    async def _analyze_concentration(self, calculations: List[ExposureCalculation]) -> Dict[str, Any]:
        """Analyze concentration across deposit types"""
        
        total_insured = sum(float(calc.insured_amount) for calc in calculations)
        
        if total_insured == 0:
            return {}
        
        # Herfindahl-Hirschman Index for concentration
        shares = [(float(calc.insured_amount) / total_insured) ** 2 for calc in calculations]
        hhi_index = sum(shares) * 10000
        
        # Concentration interpretation
        concentration_level = "LOW"
        if hhi_index > 2500:
            concentration_level = "HIGH"
        elif hhi_index > 1500:
            concentration_level = "MEDIUM"
        
        # Largest exposure
        largest_exposure = max(calculations, key=lambda x: float(x.insured_amount))
        
        return {
            "hhi_index": hhi_index,
            "concentration_level": concentration_level,
            "largest_exposure_type": largest_exposure.deposit_type.value,
            "largest_exposure_share": float(largest_exposure.insured_amount) / total_insured * 100,
            "top3_exposure_share": sum(
                sorted([float(calc.insured_amount) for calc in calculations], reverse=True)[:3]
            ) / total_insured * 100
        }