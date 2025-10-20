import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
from app.models.premiums import PremiumCalculation, PremiumStatus, CalculationMethod
from app.models.returns import ReturnPeriod, Institution
from app.models.surveillance import DepositAnalysis
from app.services.risk_scoring import RiskScoringModel
from app.schemas.premiums import PremiumCalculationRequest, PremiumCalculationResponse

class PremiumCalculationService:
    def __init__(self, db: Session):
        self.db = db
        self.risk_scoring_model = RiskScoringModel(db)

    async def calculate_premium(self, request: PremiumCalculationRequest) -> Dict[str, Any]:
        """Calculate premium for an institution using specified method"""
        
        # Validate inputs
        institution = self.db.query(Institution).filter(
            Institution.id == request.institution_id
        ).first()
        if not institution:
            return {"error": "Institution not found"}
        
        period = self.db.query(ReturnPeriod).filter(
            ReturnPeriod.id == request.period_id
        ).first()
        if not period:
            return {"error": "Return period not found"}
        
        # Get eligible deposits
        eligible_deposits = await self._get_eligible_deposits(request.institution_id, request.period_id)
        if not eligible_deposits:
            return {"error": "No eligible deposits found for premium calculation"}
        
        # Calculate average eligible deposits
        average_deposits = await self._calculate_average_eligible_deposits(eligible_deposits)
        
        # Perform calculation based on method
        if request.calculation_method == CalculationMethod.FLAT_RATE:
            calculation_result = await self._calculate_flat_rate_premium(
                average_deposits, request.base_premium_rate
            )
        else:  # RISK_BASED
            calculation_result = await self._calculate_risk_based_premium(
                request.institution_id, request.period_id, average_deposits, request.base_premium_rate
            )
        
        # Create premium calculation record
        premium_calc = PremiumCalculation(
            id=str(uuid.uuid4()),
            institution_id=request.institution_id,
            period_id=request.period_id,
            calculation_method=request.calculation_method,
            total_eligible_deposits=Decimal(str(eligible_deposits["total_deposits"])),
            average_eligible_deposits=Decimal(str(average_deposits)),
            base_premium_rate=Decimal(str(request.base_premium_rate)),
            risk_adjustment_factor=Decimal(str(calculation_result["risk_adjustment"])),
            risk_premium_rate=Decimal(str(calculation_result["premium_rate"])),
            calculated_premium=Decimal(str(calculation_result["calculated_premium"])),
            final_premium=Decimal(str(calculation_result["final_premium"])),
            calculated_by="system"  # In production, would be actual user ID
        )
        
        self.db.add(premium_calc)
        self.db.commit()
        self.db.refresh(premium_calc)
        
        return {
            "premium_calculation": PremiumCalculationResponse(
                id=premium_calc.id,
                institution_id=premium_calc.institution_id,
                period_id=premium_calc.period_id,
                calculation_method=premium_calc.calculation_method,
                total_eligible_deposits=float(premium_calc.total_eligible_deposits),
                average_eligible_deposits=float(premium_calc.average_eligible_deposits),
                base_premium_rate=float(premium_calc.base_premium_rate),
                risk_adjustment_factor=float(premium_calc.risk_adjustment_factor),
                risk_premium_rate=float(premium_calc.risk_premium_rate),
                calculated_premium=float(premium_calc.calculated_premium),
                final_premium=float(premium_calc.final_premium),
                status=premium_calc.status,
                calculated_at=premium_calc.calculated_at
            ),
            "calculation_details": calculation_result["details"]
        }
    
    async def _get_eligible_deposits(self, institution_id: str, period_id: str) -> Optional[Dict[str, Any]]:
        """Get eligible deposits for premium calculation"""
        
        # Get deposit analysis for the period
        deposit_analyses = self.db.query(DepositAnalysis).filter(
            DepositAnalysis.period_id == period_id
        ).all()
        
        if not deposit_analyses:
            return None
        
        total_deposits = sum(float(analysis.total_deposits) for analysis in deposit_analyses)
        
        # Exclude ineligible deposits (government, certain corporate, etc.)
        # This would be based on regulatory rules
        eligible_deposit_types = ["INDIVIDUAL", "CORPORATE", "JOINT", "TRUST"]
        eligible_analyses = [
            analysis for analysis in deposit_analyses 
            if analysis.deposit_type.value in eligible_deposit_types
        ]
        
        eligible_deposits = sum(float(analysis.total_deposits) for analysis in eligible_analyses)
        
        return {
            "total_deposits": total_deposits,
            "eligible_deposits": eligible_deposits,
            "ineligible_deposits": total_deposits - eligible_deposits,
            "eligible_percentage": (eligible_deposits / total_deposits * 100) if total_deposits > 0 else 0
        }
    
    async def _calculate_average_eligible_deposits(self, eligible_deposits: Dict[str, Any]) -> float:
        """Calculate arithmetic average of eligible deposits"""
        
        # In a real implementation, this would calculate the average over the period
        # For now, we'll use the eligible deposits as the average
        return eligible_deposits["eligible_deposits"]
    
    async def _calculate_flat_rate_premium(self, average_deposits: float, 
                                         base_rate: float) -> Dict[str, Any]:
        """Calculate premium using flat rate method"""
        
        calculated_premium = average_deposits * base_rate
        
        return {
            "premium_rate": base_rate,
            "risk_adjustment": 0.0,
            "calculated_premium": calculated_premium,
            "final_premium": calculated_premium,
            "details": {
                "method": "FLAT_RATE",
                "average_deposits": average_deposits,
                "premium_rate": base_rate,
                "calculation_formula": "Average Deposits × Base Rate"
            }
        }
    
    async def _calculate_risk_based_premium(self, institution_id: str, period_id: str,
                                          average_deposits: float, base_rate: float) -> Dict[str, Any]:
        """Calculate premium using risk-based method"""
        
        # Get risk score for the institution
        risk_score_result = self.risk_scoring_model.calculate_comprehensive_risk_score(
            institution_id, period_id
        )
        
        # Calculate risk adjustment factor
        risk_adjustment = await self._calculate_risk_adjustment(risk_score_result)
        
        # Apply risk adjustment to base rate
        risk_premium_rate = base_rate * (1 + risk_adjustment)
        
        # Calculate premium
        calculated_premium = average_deposits * risk_premium_rate
        
        # Apply caps and floors if needed
        final_premium = await self._apply_premium_limits(calculated_premium, average_deposits)
        
        return {
            "premium_rate": risk_premium_rate,
            "risk_adjustment": risk_adjustment,
            "calculated_premium": calculated_premium,
            "final_premium": final_premium,
            "details": {
                "method": "RISK_BASED",
                "average_deposits": average_deposits,
                "base_rate": base_rate,
                "risk_adjustment_factor": risk_adjustment,
                "risk_premium_rate": risk_premium_rate,
                "composite_risk_score": risk_score_result["risk_metrics"]["composite_risk_score"],
                "calculation_formula": "Average Deposits × Base Rate × (1 + Risk Adjustment)",
                "risk_components": risk_score_result
            }
        }
    
    async def _calculate_risk_adjustment(self, risk_score_result: Dict[str, Any]) -> float:
        """Calculate risk adjustment factor based on risk score"""
        
        composite_score = risk_score_result["risk_metrics"]["composite_risk_score"]
        
        # Risk adjustment curve - higher risk = higher adjustment
        if composite_score < 0.2:
            return -0.2  # 20% discount for low risk
        elif composite_score < 0.4:
            return -0.1  # 10% discount for moderate-low risk
        elif composite_score < 0.6:
            return 0.0   # No adjustment for medium risk
        elif composite_score < 0.8:
            return 0.3   # 30% surcharge for moderate-high risk
        else:
            return 0.6   # 60% surcharge for high risk
    
    async def _apply_premium_limits(self, premium: float, average_deposits: float) -> float:
        """Apply regulatory limits to premium calculation"""
        
        # Minimum premium
        min_premium = 1000.00  # $1,000 minimum
        premium = max(premium, min_premium)
        
        # Maximum premium as percentage of deposits
        max_premium_ratio = 0.01  # 1% of deposits maximum
        max_premium = average_deposits * max_premium_ratio
        premium = min(premium, max_premium)
        
        return premium
    
    async def recalculate_premium(self, calculation_id: str, new_rate: Optional[float] = None) -> Dict[str, Any]:
        """Recalculate premium with updated parameters"""
        
        existing_calc = self.db.query(PremiumCalculation).filter(
            PremiumCalculation.id == calculation_id
        ).first()
        
        if not existing_calc:
            return {"error": "Premium calculation not found"}
        
        if existing_calc.status != PremiumStatus.CALCULATED:
            return {"error": "Cannot recalculate premium that is already invoiced"}
        
        # Use new rate if provided, otherwise use existing rate
        base_rate = Decimal(str(new_rate)) if new_rate else existing_calc.base_premium_rate
        
        if existing_calc.calculation_method == CalculationMethod.FLAT_RATE:
            new_premium = existing_calc.average_eligible_deposits * base_rate
            risk_adjustment = Decimal('0.0')
            risk_rate = base_rate
        else:
            # Recalculate risk-based premium
            risk_score_result = self.risk_scoring_model.calculate_comprehensive_risk_score(
                existing_calc.institution_id, existing_calc.period_id
            )
            risk_adjustment = Decimal(str(await self._calculate_risk_adjustment(risk_score_result)))
            risk_rate = base_rate * (1 + risk_adjustment)
            new_premium = existing_calc.average_eligible_deposits * risk_rate
        
        # Update calculation
        existing_calc.base_premium_rate = base_rate
        existing_calc.risk_adjustment_factor = risk_adjustment
        existing_calc.risk_premium_rate = risk_rate
        existing_calc.calculated_premium = new_premium
        existing_calc.final_premium = new_premium
        existing_calc.calculated_at = datetime.utcnow()
        
        self.db.commit()
        
        return {
            "recalculated_premium": float(new_premium),
            "previous_premium": float(existing_calc.calculated_premium),
            "difference": float(new_premium - existing_calc.calculated_premium)
        }