import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
from models import (
    DepositAnalysis,
    SurveillancePeriod,
    DepositType,
    AccountSize,
    ReturnUpload,
    ReturnPeriod,
    Institution,
    ReturnStatus,
    FileType,
)

class DepositAnalysisService:
    def __init__(self, db: Session):
        self.db = db
        
    async def analyze_deposits(self, institution_id: str, period_type: str,
                              period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Comprehensive deposit analysis for surveillance"""
        
        # Create surveillance period
        period = await self._create_surveillance_period(
            institution_id, period_type, period_start, period_end
        )
        
        # Get deposit data from returns
        deposit_data = await self._extract_deposit_data(institution_id, period_start, period_end)
        
        if deposit_data is None or deposit_data.empty:
            return {"error": "No deposit data available for analysis"}
        
        analyses = []
        
        # Analyze by deposit type
        for deposit_type in DepositType:
            type_analysis = await self._analyze_deposit_type(
                deposit_data, deposit_type, period.id
            )
            if type_analysis:
                analyses.append(type_analysis)
        
        # Calculate trends
        trends = await self._calculate_deposit_trends(institution_id, period_start, period_end)
        
        # Generate insights
        insights = await self._generate_deposit_insights(analyses, trends)
        
        return {
            "period": {
                "id": period.id,
                "type": period_type,
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            },
            "analyses": analyses,
            "trends": trends,
            "insights": insights,
            "summary_metrics": await self._calculate_summary_metrics(analyses),
            "analysis_date": datetime.utcnow().isoformat()
        }
    
    async def _extract_deposit_data(self, institution_id: str, 
                                  period_start: datetime, period_end: datetime) -> Optional[pd.DataFrame]:
        """Extract deposit data from submitted returns"""
        
        # Find the relevant return period
        return_period = self.db.query(ReturnPeriod).filter(
            ReturnPeriod.institution_id == institution_id,
            ReturnPeriod.period_start == period_start,
            ReturnPeriod.period_end == period_end,
            ReturnPeriod.status == ReturnStatus.SUBMITTED
        ).first()
        
        if not return_period:
            return None
        
        # Get deposit upload
        deposit_upload = self.db.query(ReturnUpload).filter(
            ReturnUpload.period_id == return_period.id,
            ReturnUpload.file_type == FileType.DEPOSIT
        ).first()
        
        if not deposit_upload:
            return None
        
        # Read deposit file
        try:
            if deposit_upload.file_path.endswith('.csv'):
                df = pd.read_csv(deposit_upload.file_path)
            else:
                df = pd.read_excel(deposit_upload.file_path)
            
            return df
        except Exception as e:
            print(f"Error reading deposit file: {e}")
            return None
    
    async def _analyze_deposit_type(self, deposit_data: pd.DataFrame, 
                                  deposit_type: DepositType, period_id: str) -> Dict[str, Any]:
        """Analyze deposits for a specific type"""
        
        # Filter data by deposit type (this would use actual classification logic)
        if deposit_type == DepositType.INDIVIDUAL:
            type_data = deposit_data[deposit_data['account_type'].isin(['SAVINGS', 'CHECKING', 'INDIVIDUAL'])]
        elif deposit_type == DepositType.CORPORATE:
            type_data = deposit_data[deposit_data['account_type'].isin(['CORPORATE', 'BUSINESS'])]
        elif deposit_type == DepositType.GOVERNMENT:
            type_data = deposit_data[deposit_data['account_type'].isin(['GOVERNMENT'])]
        else:
            type_data = deposit_data[deposit_data['account_type'] == deposit_type.value]
        
        if len(type_data) == 0:
            return None
        
        total_deposits = type_data['balance'].sum()
        total_accounts = len(type_data)
        average_balance = total_deposits / total_accounts if total_accounts > 0 else 0
        
        # Calculate breakdowns
        currency_breakdown = self._calculate_currency_breakdown(type_data)
        account_size_breakdown = self._calculate_account_size_breakdown(type_data)
        product_breakdown = self._calculate_product_breakdown(type_data)
        
        # Calculate growth (would compare with previous period)
        growth_rate = await self._calculate_growth_rate(deposit_type, total_deposits, period_id)
        
        # Create analysis record
        analysis = DepositAnalysis(
            id=str(uuid.uuid4()),
            period_id=period_id,
            deposit_type=deposit_type,
            total_deposits=Decimal(str(total_deposits)),
            total_accounts=total_accounts,
            average_balance=Decimal(str(average_balance)),
            growth_rate=Decimal(str(growth_rate)) if growth_rate else None,
            currency_breakdown=currency_breakdown,
            account_size_breakdown=account_size_breakdown,
            product_breakdown=product_breakdown
        )
        
        self.db.add(analysis)
        self.db.commit()
        
        return {
            "deposit_type": deposit_type.value,
            "total_deposits": float(total_deposits),
            "total_accounts": total_accounts,
            "average_balance": float(average_balance),
            "growth_rate": float(growth_rate) if growth_rate else None,
            "currency_breakdown": currency_breakdown,
            "account_size_breakdown": account_size_breakdown,
            "product_breakdown": product_breakdown,
            "market_share": await self._calculate_market_share(deposit_type, total_deposits)
        }
    
    def _calculate_currency_breakdown(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate currency distribution"""
        if 'currency' not in data.columns:
            return {}
        
        currency_totals = data.groupby('currency')['balance'].sum()
        return {currency: float(amount) for currency, amount in currency_totals.items()}
    
    def _calculate_account_size_breakdown(self, data: pd.DataFrame) -> Dict[str, int]:
        """Classify accounts by size"""
        if 'balance' not in data.columns:
            return {}
        
        def classify_account_size(balance):
            if balance < 10000:
                return AccountSize.SMALL.value
            elif balance < 100000:
                return AccountSize.MEDIUM.value
            else:
                return AccountSize.LARGE.value
        
        data['size_category'] = data['balance'].apply(classify_account_size)
        size_counts = data['size_category'].value_counts().to_dict()
        
        return size_counts
    
    def _calculate_product_breakdown(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate product type distribution"""
        if 'account_type' not in data.columns:
            return {}
        
        product_totals = data.groupby('account_type')['balance'].sum()
        return {product: float(amount) for product, amount in product_totals.items()}
    
    async def _calculate_growth_rate(self, deposit_type: DepositType, 
                                   current_total: float, period_id: str) -> Optional[float]:
        """Calculate growth rate from previous period"""
        # This would query previous period's data
        # For now, return a mock growth rate
        import random
        return random.uniform(-0.05, 0.15)  # -5% to +15%
    
    async def _calculate_market_share(self, deposit_type: DepositType, 
                                    institution_deposits: float) -> float:
        """Calculate market share for this deposit type"""
        # This would query industry-wide totals
        # For now, return a mock market share
        industry_totals = {
            DepositType.INDIVIDUAL: 5000000000,  # $5B
            DepositType.CORPORATE: 3000000000,   # $3B
            DepositType.GOVERNMENT: 1000000000,  # $1B
            DepositType.JOINT: 500000000,        # $500M
            DepositType.TRUST: 300000000         # $300M
        }
        
        industry_total = industry_totals.get(deposit_type, 1000000000)
        return (institution_deposits / industry_total) * 100
    
    async def _calculate_deposit_trends(self, institution_id: str, 
                                      period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Calculate deposit trends over time"""
        
        # Get historical data (last 12 months)
        historical_periods = self.db.query(SurveillancePeriod).filter(
            SurveillancePeriod.institution_id == institution_id,
            SurveillancePeriod.period_end <= period_end,
            SurveillancePeriod.period_end >= period_end - timedelta(days=365)
        ).order_by(SurveillancePeriod.period_end).all()
        
        trends = {
            "total_deposits": [],
            "account_growth": [],
            "composition_changes": [],
            "volatility_metrics": {}
        }
        
        for period in historical_periods:
            analyses = self.db.query(DepositAnalysis).filter(
                DepositAnalysis.period_id == period.id
            ).all()
            
            total_deposits = sum(float(analysis.total_deposits) for analysis in analyses)
            total_accounts = sum(analysis.total_accounts for analysis in analyses)
            
            trends["total_deposits"].append({
                "period": period.period_end.isoformat(),
                "value": total_deposits
            })
            
            trends["account_growth"].append({
                "period": period.period_end.isoformat(),
                "value": total_accounts
            })
        
        # Calculate volatility
        if len(trends["total_deposits"]) > 1:
            deposits_series = [item["value"] for item in trends["total_deposits"]]
            trends["volatility_metrics"] = {
                "std_deviation": np.std(deposits_series),
                "coefficient_of_variation": np.std(deposits_series) / np.mean(deposits_series),
                "max_drawdown": self._calculate_max_drawdown(deposits_series)
            }
        
        return trends
    
    def _calculate_max_drawdown(self, series: List[float]) -> float:
        """Calculate maximum drawdown from peak"""
        peak = series[0]
        max_drawdown = 0
        
        for value in series[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    async def _generate_deposit_insights(self, analyses: List[Dict], 
                                       trends: Dict) -> List[Dict[str, Any]]:
        """Generate actionable insights from deposit analysis"""
        
        insights = []
        
        # Growth insights
        total_deposits = sum(analysis["total_deposits"] for analysis in analyses)
        corporate_deposits = next((a for a in analyses if a["deposit_type"] == "CORPORATE"), None)
        
        if corporate_deposits and corporate_deposits.get("growth_rate", 0) > 0.1:
            insights.append({
                "type": "POSITIVE",
                "category": "GROWTH",
                "title": "Strong Corporate Deposit Growth",
                "description": f"Corporate deposits growing at {corporate_deposits['growth_rate']:.1%}",
                "impact": "MEDIUM",
                "recommendation": "Monitor concentration risks"
            })
        
        # Concentration insights
        largest_type = max(analyses, key=lambda x: x["total_deposits"])
        if largest_type["total_deposits"] / total_deposits > 0.6:
            insights.append({
                "type": "WARNING",
                "category": "CONCENTRATION",
                "title": "High Deposit Concentration",
                "description": f"{largest_type['deposit_type']} deposits represent {largest_type['total_deposits']/total_deposits:.1%} of total",
                "impact": "HIGH",
                "recommendation": "Diversify deposit base"
            })
        
        # Volatility insights
        if trends.get("volatility_metrics", {}).get("coefficient_of_variation", 0) > 0.15:
            insights.append({
                "type": "WARNING",
                "category": "VOLATILITY",
                "title": "High Deposit Volatility",
                "description": "Deposit base shows significant fluctuations",
                "impact": "MEDIUM",
                "recommendation": "Review funding stability"
            })
        
        return insights
    
    async def _calculate_summary_metrics(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Calculate summary metrics for all deposit types"""
        
        total_deposits = sum(analysis["total_deposits"] for analysis in analyses)
        total_accounts = sum(analysis["total_accounts"] for analysis in analyses)
        
        return {
            "total_deposits": total_deposits,
            "total_accounts": total_accounts,
            "average_balance": total_deposits / total_accounts if total_accounts > 0 else 0,
            "deposit_composition": {
                analysis["deposit_type"]: analysis["total_deposits"] / total_deposits 
                for analysis in analyses
            },
            "growth_composition": {
                analysis["deposit_type"]: analysis.get("growth_rate", 0)
                for analysis in analyses
            }
        }
    
    async def _create_surveillance_period(self, institution_id: str, period_type: str,
                                        period_start: datetime, period_end: datetime) -> SurveillancePeriod:
        """Create surveillance period record"""
        
        period = SurveillancePeriod(
            id=str(uuid.uuid4()),
            institution_id=institution_id,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end
        )
        
        self.db.add(period)
        self.db.commit()
        self.db.refresh(period)
        
        return period