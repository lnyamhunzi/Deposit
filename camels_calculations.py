# app/services/camels_calculations.py
from sqlalchemy.orm import Session
from decimal import Decimal
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class CAMELSCalculations:
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_camels_ratings(self, institution_id: str, period_id: str, financial_data: Dict) -> Dict:
        """Calculate complete CAMELS ratings"""
        
        # Calculate individual components
        capital_adequacy = self._calculate_capital_adequacy(financial_data)
        asset_quality = self._calculate_asset_quality(financial_data)
        management_quality = self._calculate_management_quality(financial_data)
        earnings = self._calculate_earnings(financial_data)
        liquidity = self._calculate_liquidity(financial_data)
        sensitivity = self._calculate_sensitivity(financial_data)
        
        # Calculate composite rating
        composite_rating = self._calculate_composite_rating(
            capital_adequacy, asset_quality, management_quality, 
            earnings, liquidity, sensitivity
        )
        
        return {
            "capital_adequacy": {
                "score": capital_adequacy["score"],
                "rating": capital_adequacy["rating"],
                "components": capital_adequacy["components"]
            },
            "asset_quality": {
                "score": asset_quality["score"],
                "rating": asset_quality["rating"],
                "components": asset_quality["components"]
            },
            "management_quality": {
                "score": management_quality["score"],
                "rating": management_quality["rating"],
                "components": management_quality["components"]
            },
            "earnings": {
                "score": earnings["score"],
                "rating": earnings["rating"],
                "components": earnings["components"]
            },
            "liquidity": {
                "score": liquidity["score"],
                "rating": liquidity["rating"],
                "components": liquidity["components"]
            },
            "sensitivity": {
                "score": sensitivity["score"],
                "rating": sensitivity["rating"],
                "components": sensitivity["components"]
            },
            "composite_rating": composite_rating,
            "risk_grade": self._get_risk_grade(composite_rating),
            "calculated_at": datetime.utcnow()
        }
    
    def _calculate_capital_adequacy(self, data: Dict) -> Dict:
        """Calculate Capital Adequacy component"""
        
        # Capital Adequacy Ratio (CAR)
        tier1_capital = data.get('tier1_capital', Decimal('0'))
        tier2_capital = data.get('tier2_capital', Decimal('0'))
        risk_weighted_assets = data.get('risk_weighted_assets', Decimal('1'))
        
        car = ((tier1_capital + tier2_capital) / risk_weighted_assets) * 100
        
        # Tier 1 Capital Ratio
        tier1_ratio = (tier1_capital / risk_weighted_assets) * 100

        # Tier 2 Capital Ratio
        tier2_ratio = (tier2_capital / risk_weighted_assets) * 100
        
        # Capital to Assets Ratio
        total_assets = data.get('total_assets', Decimal('1'))
        capital_to_assets = ((tier1_capital + tier2_capital) / total_assets) * 100
        
        # Equity to Assets Ratio
        total_equity = data.get('total_equity', Decimal('0'))
        equity_to_assets = (total_equity / total_assets) * 100
        
        # Score calculation with regulatory thresholds
        car_score = self._rate_capital_ratio(car, min_threshold=10, strong_threshold=15)
        tier1_score = self._rate_capital_ratio(tier1_ratio, min_threshold=6, strong_threshold=8.5)
        capital_assets_score = self._rate_capital_ratio(capital_to_assets, min_threshold=8, strong_threshold=12)
        
        component_score = (car_score + tier1_score + capital_assets_score) / 3
        
        return {
            "score": float(component_score),
            "rating": self._get_component_rating(component_score),
            "components": {
                "capital_adequacy_ratio": float(car),
                "tier1_ratio": float(tier1_ratio),
                "tier2_ratio": float(tier2_ratio),
                "capital_to_assets": float(capital_to_assets),
                "equity_to_assets": float(equity_to_assets)
            }
        }
    
    def _calculate_asset_quality(self, data: Dict) -> Dict:
        """Calculate Asset Quality component"""
        
        total_assets = data.get('total_assets', Decimal('1'))
        gross_npa = data.get('gross_npa', Decimal('0'))
        net_npa = data.get('net_npa', Decimal('0'))
        total_loans = data.get('total_loans', Decimal('1'))
        provisions = data.get('provisions', Decimal('0'))
        top_10_exposures = data.get('top_10_exposures', Decimal('0'))
        
        # NPA Ratios
        gross_npa_ratio = (gross_npa / total_loans) * 100
        net_npa_ratio = (net_npa / total_loans) * 100
        
        # Provision Coverage Ratio
        provision_coverage = (provisions / gross_npa) * 100 if gross_npa > 0 else 100
        
        # Loan Loss Reserve Ratio
        loan_loss_reserve = (provisions / total_loans) * 100

        # Asset Concentration
        asset_concentration = (top_10_exposures / total_loans) * 100 if total_loans > 0 else 0
        
        # Asset Growth Rate (year-over-year)
        previous_assets = data.get('previous_year_assets', total_assets)
        asset_growth = ((total_assets - previous_assets) / previous_assets) * 100
        
        # Score calculation
        npa_score = self._rate_npa_ratio(net_npa_ratio, max_threshold=5, good_threshold=2)
        coverage_score = self._rate_coverage_ratio(provision_coverage, min_threshold=70, strong_threshold=85)
        growth_score = self._rate_growth_ratio(asset_growth, optimal_range=(8, 15))
        
        component_score = (npa_score * 0.5 + coverage_score * 0.3 + growth_score * 0.2)
        
        return {
            "score": float(component_score),
            "rating": self._get_component_rating(component_score),
            "components": {
                "gross_npa_ratio": float(gross_npa_ratio),
                "net_npa_ratio": float(net_npa_ratio),
                "provision_coverage": float(provision_coverage),
                "loan_loss_reserve": float(loan_loss_reserve),
                "asset_concentration": float(asset_concentration),
                "asset_growth": float(asset_growth)
            }
        }
    
    def _calculate_management_quality(self, data: Dict) -> Dict:
        """Calculate Management Quality component"""
        
        total_income = data.get('total_income', Decimal('1'))
        operating_expenses = data.get('operating_expenses', Decimal('0'))
        total_assets = data.get('total_assets', Decimal('1'))
        total_equity = data.get('total_equity', Decimal('1'))
        
        # Cost to Income Ratio
        cost_to_income = (operating_expenses / total_income) * 100
        
        # Asset Utilization
        asset_utilization = (total_income / total_assets) * 100
        
        # Business per Employee (simplified)
        total_employees = data.get('total_employees', 1)
        business_per_employee = total_assets / total_employees
        
        # Return on Equity (for management efficiency)
        net_income = data.get('net_income', Decimal('0'))
        roe = (net_income / total_equity) * 100
        
        # Score calculation
        cost_score = self._rate_cost_ratio(cost_to_income, max_threshold=60, good_threshold=45)
        asset_util_score = self._rate_utilization_ratio(asset_utilization, min_threshold=2, strong_threshold=3)
        roe_score = self._rate_roe_ratio(roe, min_threshold=8, strong_threshold=12)
        
        component_score = (cost_score * 0.4 + asset_util_score * 0.3 + roe_score * 0.3)
        
        return {
            "score": float(component_score),
            "rating": self._get_component_rating(component_score),
            "components": {
                "cost_to_income": float(cost_to_income),
                "asset_utilization": float(asset_utilization),
                "business_per_employee": float(business_per_employee),
                "return_on_equity": float(roe)
            }
        }
    
    def _calculate_earnings(self, data: Dict) -> Dict:
        """Calculate Earnings component"""
        
        net_income = data.get('net_income', Decimal('0'))
        total_assets = data.get('total_assets', Decimal('1'))
        total_equity = data.get('total_equity', Decimal('1'))
        operating_income = data.get('operating_income', Decimal('1'))
        interest_income = data.get('interest_income', Decimal('0'))
        interest_expense = data.get('interest_expense', Decimal('0'))
        earnings_growth = data.get('earnings_growth', Decimal('0'))
        
        # Return on Assets (ROA)
        roa = (net_income / total_assets) * 100
        
        # Return on Equity (ROE)
        roe = (net_income / total_equity) * 100
        
        # Net Interest Margin (NIM)
        earning_assets = data.get('earning_assets', total_assets)
        nim = ((interest_income - interest_expense) / earning_assets) * 100
        
        # Operating Profit Margin
        operating_margin = (operating_income / total_assets) * 100
        
        # Income Diversity
        non_interest_income = data.get('non_interest_income', Decimal('0'))
        income_diversity = (non_interest_income / operating_income) * 100 if operating_income > 0 else 0

        # Earnings Trend
        earnings_trend = earnings_growth
        
        # Score calculation
        roa_score = self._rate_roa_ratio(roa, min_threshold=0.5, strong_threshold=1.2)
        roe_score = self._rate_roe_ratio(roe, min_threshold=8, strong_threshold=15)
        nim_score = self._rate_nim_ratio(nim, min_threshold=2, strong_threshold=3.5)
        
        component_score = (roa_score * 0.4 + roe_score * 0.3 + nim_score * 0.3)
        
        return {
            "score": float(component_score),
            "rating": self._get_component_rating(component_score),
            "components": {
                "return_on_assets": float(roa),
                "return_on_equity": float(roe),
                "net_interest_margin": float(nim),
                "operating_margin": float(operating_margin),
                "income_diversity": float(income_diversity),
                "earnings_trend": float(earnings_trend)
            }
        }
    
    def _calculate_liquidity(self, data: Dict) -> Dict:
        """Calculate Liquidity component"""
        
        liquid_assets = data.get('liquid_assets', Decimal('0'))
        cash_equivalents = data.get('cash_equivalents', Decimal('0'))
        total_assets = data.get('total_assets', Decimal('1'))
        short_term_liabilities = data.get('short_term_liabilities', Decimal('1'))
        total_deposits = data.get('total_deposits', Decimal('1'))
        total_loans = data.get('total_loans', Decimal('0'))
        
        # Liquidity Ratio
        liquidity_ratio = (liquid_assets / total_assets) * 100

        # Cash to Assets Ratio
        cash_to_assets_ratio = (cash_equivalents / total_assets) * 100
        
        # Loan to Deposit Ratio
        loan_to_deposit = (total_loans / total_deposits) * 100
        
        # Quick Ratio
        quick_ratio = (liquid_assets / short_term_liabilities) * 100
        
        # Core Deposits Ratio
        core_deposits = data.get('core_deposits', total_deposits * Decimal('0.7'))
        core_deposits_ratio = (core_deposits / total_deposits) * 100
        
        # Score calculation
        liquidity_score = self._rate_liquidity_ratio(liquidity_ratio, min_threshold=20, strong_threshold=30)
        ltd_score = self._rate_ltd_ratio(loan_to_deposit, max_threshold=80, good_threshold=75)
        quick_score = self._rate_quick_ratio(quick_ratio, min_threshold=100, strong_threshold=120)
        
        component_score = (liquidity_score * 0.4 + ltd_score * 0.4 + quick_score * 0.2)
        
        return {
            "score": float(component_score),
            "rating": self._get_component_rating(component_score),
            "components": {
                "liquidity_ratio": float(liquidity_ratio),
                "cash_to_assets_ratio": float(cash_to_assets_ratio),
                "loan_to_deposit": float(loan_to_deposit),
                "quick_ratio": float(quick_ratio),
                "core_deposits_ratio": float(core_deposits_ratio)
            }
        }
    
    def _calculate_sensitivity(self, data: Dict) -> Dict:
        """Calculate Sensitivity to Market Risk component"""
        
        total_assets = data.get('total_assets', Decimal('1'))
        foreign_currency_assets = data.get('foreign_currency_assets', Decimal('0'))
        foreign_currency_liabilities = data.get('foreign_currency_liabilities', Decimal('0'))
        interest_sensitive_assets = data.get('interest_sensitive_assets', Decimal('0'))
        interest_sensitive_liabilities = data.get('interest_sensitive_liabilities', Decimal('0'))
        fx_exposure = data.get('fx_exposure', Decimal('0'))
        
        # Foreign Exchange Exposure
        fx_risk = (abs(foreign_currency_assets - foreign_currency_liabilities) / total_assets) * 100
        
        # Interest Rate Sensitivity Gap
        interest_gap = (interest_sensitive_assets - interest_sensitive_liabilities) / total_assets * 100
        
        # Concentration Risk (simplified)
        largest_depositor = data.get('largest_depositor', Decimal('0'))
        deposit_concentration = (largest_depositor / total_assets) * 100

        # Market Concentration
        market_concentration = fx_exposure
        
        # Volatility of Earnings
        current_earnings = data.get('net_income', Decimal('0'))
        previous_earnings = data.get('previous_net_income', current_earnings)
        earnings_volatility = abs((current_earnings - previous_earnings) / previous_earnings * 100) if previous_earnings > 0 else 0
        
        # Score calculation
        fx_score = self._rate_fx_exposure(abs(fx_risk), max_threshold=10, good_threshold=5)
        interest_score = self._rate_interest_gap(abs(interest_gap), max_threshold=15, good_threshold=10)
        concentration_score = self._rate_concentration_ratio(deposit_concentration, max_threshold=20, good_threshold=15)
        
        component_score = (fx_score * 0.4 + interest_score * 0.3 + concentration_score * 0.3)
        
        return {
            "score": float(component_score),
            "rating": self._get_component_rating(component_score),
            "components": {
                "fx_risk": float(fx_risk),
                "interest_rate_gap": float(interest_gap),
                "deposit_concentration": float(deposit_concentration),
                "market_concentration": float(market_concentration),
                "earnings_volatility": float(earnings_volatility)
            }
        }
    
    def _calculate_composite_rating(self, capital: Dict, assets: Dict, management: Dict, 
                                  earnings: Dict, liquidity: Dict, sensitivity: Dict) -> float:
        """Calculate composite CAMELS rating with weights"""
        
        # Regulatory weights (can be adjusted)
        weights = {
            'capital': 0.25,
            'assets': 0.20,
            'management': 0.25,
            'earnings': 0.10,
            'liquidity': 0.10,
            'sensitivity': 0.10
        }
        
        composite = (
            capital['score'] * weights['capital'] +
            assets['score'] * weights['assets'] +
            management['score'] * weights['management'] +
            earnings['score'] * weights['earnings'] +
            liquidity['score'] * weights['liquidity'] +
            sensitivity['score'] * weights['sensitivity']
        )
        
        return float(composite)
    
    # Rating helper methods
    def _rate_capital_ratio(self, ratio: Decimal, min_threshold: float, strong_threshold: float) -> float:
        if ratio >= strong_threshold:
            return 1.0  # Strong
        elif ratio >= min_threshold:
            return 2.0  # Satisfactory
        elif ratio >= min_threshold - 2:
            return 3.0  # Fair
        elif ratio >= min_threshold - 4:
            return 4.0  # Marginal
        else:
            return 5.0  # Unsatisfactory
    
    def _rate_npa_ratio(self, ratio: Decimal, max_threshold: float, good_threshold: float) -> float:
        if ratio <= good_threshold:
            return 1.0
        elif ratio <= max_threshold:
            return 2.0
        elif ratio <= max_threshold + 3:
            return 3.0
        elif ratio <= max_threshold + 6:
            return 4.0
        else:
            return 5.0
    
    def _rate_coverage_ratio(self, ratio: Decimal, min_threshold: float, strong_threshold: float) -> float:
        if ratio >= strong_threshold:
            return 1.0
        elif ratio >= min_threshold:
            return 2.0
        elif ratio >= min_threshold - 15:
            return 3.0
        elif ratio >= min_threshold - 30:
            return 4.0
        else:
            return 5.0
    
    def _rate_cost_ratio(self, ratio: Decimal, max_threshold: float, good_threshold: float) -> float:
        if ratio <= good_threshold:
            return 1.0
        elif ratio <= max_threshold:
            return 2.0
        elif ratio <= max_threshold + 10:
            return 3.0
        elif ratio <= max_threshold + 20:
            return 4.0
        else:
            return 5.0
    
    def _rate_roa_ratio(self, ratio: Decimal, min_threshold: float, strong_threshold: float) -> float:
        if ratio >= strong_threshold:
            return 1.0
        elif ratio >= min_threshold:
            return 2.0
        elif ratio >= 0:
            return 3.0
        elif ratio >= -min_threshold:
            return 4.0
        else:
            return 5.0
    
    def _rate_liquidity_ratio(self, ratio: Decimal, min_threshold: float, strong_threshold: float) -> float:
        if ratio >= strong_threshold:
            return 1.0
        elif ratio >= min_threshold:
            return 2.0
        elif ratio >= min_threshold - 5:
            return 3.0
        elif ratio >= min_threshold - 10:
            return 4.0
        else:
            return 5.0
    
    def _rate_ltd_ratio(self, ratio: Decimal, max_threshold: float, good_threshold: float) -> float:
        if ratio <= good_threshold:
            return 1.0
        elif ratio <= max_threshold:
            return 2.0
        elif ratio <= max_threshold + 10:
            return 3.0
        elif ratio <= max_threshold + 20:
            return 4.0
        else:
            return 5.0
    
    def _get_component_rating(self, score: float) -> str:
        if score <= 1.5:
            return "STRONG"
        elif score <= 2.5:
            return "SATISFACTORY"
        elif score <= 3.5:
            return "FAIR"
        elif score <= 4.5:
            return "MARGINAL"
        else:
            return "UNSATISFACTORY"
    
    def _get_risk_grade(self, composite_score: float) -> str:
        if composite_score <= 1.8:
            return "A"  # Excellent
        elif composite_score <= 2.5:
            return "B"  # Good
        elif composite_score <= 3.2:
            return "C"  # Fair
        elif composite_score <= 4.0:
            return "D"  # Marginal
        else:
            return "E"  # Unsatisfactory