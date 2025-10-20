# app/services/stress_testing.py
from sqlalchemy.orm import Session
from decimal import Decimal
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
from scipy import stats
from camels_calculations import CAMELSCalculations

class StressTestingEngine:
    def __init__(self, db: Session):
        self.db = db
        self.scenarios = self._initialize_scenarios()
        
    def run_stress_test(self, institution_id: str, scenario_type: str, severity: str = "MODERATE") -> Dict:
        """Run comprehensive stress test"""
        
        # Get current financial data
        financial_data = self._get_financial_data(institution_id)
        
        # Get scenario parameters
        scenario = self.scenarios[scenario_type][severity]
        
        # Apply shocks
        stressed_data = self._apply_shocks(financial_data, scenario)
        
        # Calculate impact on key metrics
        impact_analysis = self._calculate_impact(financial_data, stressed_data)
        
        # Calculate new CAMELS rating under stress
        camels_calc = CAMELSCalculations(self.db)
        current_camels = camels_calc.calculate_camels_ratings(institution_id, "current", financial_data)
        stressed_camels = camels_calc.calculate_camels_ratings(institution_id, "stressed", stressed_data)
        
        # Calculate capital adequacy under stress
        capital_impact = self._calculate_capital_impact(financial_data, stressed_data)
        
        # Probability of Default under stress
        pd_stressed = self._calculate_pd_under_stress(financial_data, stressed_data)
        
        return {
            "scenario": {
                "type": scenario_type,
                "severity": severity,
                "parameters": scenario
            },
            "current_financials": financial_data,
            "stressed_financials": stressed_data,
            "impact_analysis": impact_analysis,
            "camels_comparison": {
                "current": current_camels,
                "stressed": stressed_camels,
                "deterioration": self._calculate_camels_deterioration(current_camels, stressed_camels)
            },
            "capital_adequacy_impact": capital_impact,
            "risk_metrics": {
                "probability_of_default": pd_stressed,
                "capital_shortfall": capital_impact.get('capital_shortfall', Decimal('0')),
                "liquidity_gap": self._calculate_liquidity_gap(stressed_data)
            },
            "recommendations": self._generate_recommendations(impact_analysis, stressed_camels),
            "run_date": datetime.utcnow()
        }
    
    def _initialize_scenarios(self) -> Dict:
        """Initialize stress testing scenarios"""
        
        return {
            "EXCHANGE_RATE_SHOCK": {
                "MILD": {
                    "local_currency_depreciation": Decimal('0.15'),  # 15%
                    "import_cost_increase": Decimal('0.10'),
                    "export_income_decrease": Decimal('0.08'),
                    "inflation_impact": Decimal('0.05')
                },
                "MODERATE": {
                    "local_currency_depreciation": Decimal('0.30'),  # 30%
                    "import_cost_increase": Decimal('0.20'),
                    "export_income_decrease": Decimal('0.15'),
                    "inflation_impact": Decimal('0.10')
                },
                "SEVERE": {
                    "local_currency_depreciation": Decimal('0.50'),  # 50%
                    "import_cost_increase": Decimal('0.35'),
                    "export_income_decrease": Decimal('0.25'),
                    "inflation_impact": Decimal('0.20')
                }
            },
            "LIQUIDITY_SHOCK": {
                "MILD": {
                    "deposit_withdrawal": Decimal('0.10'),  # 10% deposit withdrawal
                    "funding_cost_increase": Decimal('0.02'),  # 2% increase
                    "liquid_assets_decrease": Decimal('0.15')
                },
                "MODERATE": {
                    "deposit_withdrawal": Decimal('0.20'),  # 20% deposit withdrawal
                    "funding_cost_increase": Decimal('0.05'),  # 5% increase
                    "liquid_assets_decrease": Decimal('0.25')
                },
                "SEVERE": {
                    "deposit_withdrawal": Decimal('0.35'),  # 35% deposit withdrawal
                    "funding_cost_increase": Decimal('0.10'),  # 10% increase
                    "liquid_assets_decrease": Decimal('0.40')
                }
            },
            "INTEREST_RATE_SHOCK": {
                "MILD": {
                    "rate_increase": Decimal('0.02'),  # 200 basis points
                    "bond_portfolio_loss": Decimal('0.05'),
                    "net_interest_margin_impact": Decimal('0.01')
                },
                "MODERATE": {
                    "rate_increase": Decimal('0.05'),  # 500 basis points
                    "bond_portfolio_loss": Decimal('0.12'),
                    "net_interest_margin_impact": Decimal('0.03')
                },
                "SEVERE": {
                    "rate_increase": Decimal('0.10'),  # 1000 basis points
                    "bond_portfolio_loss": Decimal('0.25'),
                    "net_interest_margin_impact": Decimal('0.06')
                }
            },
            "MACROECONOMIC_DOWNTURN": {
                "MILD": {
                    "gdp_contraction": Decimal('0.03'),  # 3% GDP contraction
                    "unemployment_increase": Decimal('0.02'),  # 2% increase
                    "loan_default_increase": Decimal('0.05'),  # 5% increase in defaults
                    "asset_price_decline": Decimal('0.10')  # 10% asset price decline
                },
                "MODERATE": {
                    "gdp_contraction": Decimal('0.06'),  # 6% GDP contraction
                    "unemployment_increase": Decimal('0.05'),  # 5% increase
                    "loan_default_increase": Decimal('0.15'),  # 15% increase in defaults
                    "asset_price_decline": Decimal('0.25')  # 25% asset price decline
                },
                "SEVERE": {
                    "gdp_contraction": Decimal('0.10'),  # 10% GDP contraction
                    "unemployment_increase": Decimal('0.08'),  # 8% increase
                    "loan_default_increase": Decimal('0.30'),  # 30% increase in defaults
                    "asset_price_decline": Decimal('0.40')  # 40% asset price decline
                }
            },
            "SECTORAL_SHOCK": {
                "MILD": {
                    "sector_concentration_risk": Decimal('0.10'),
                    "sector_default_rate": Decimal('0.08'),
                    "collateral_value_decline": Decimal('0.15')
                },
                "MODERATE": {
                    "sector_concentration_risk": Decimal('0.25'),
                    "sector_default_rate": Decimal('0.20'),
                    "collateral_value_decline": Decimal('0.30')
                },
                "SEVERE": {
                    "sector_concentration_risk": Decimal('0.40'),
                    "sector_default_rate": Decimal('0.35'),
                    "collateral_value_decline": Decimal('0.50')
                }
            }
        }
    
    def _apply_shocks(self, financial_data: Dict, scenario: Dict) -> Dict:
        """Apply scenario shocks to financial data"""
        
        stressed_data = financial_data.copy()
        
        # Exchange Rate Shock impacts
        if 'local_currency_depreciation' in scenario:
            depreciation = scenario['local_currency_depreciation']
            
            # Impact on foreign currency assets
            fc_assets = stressed_data.get('foreign_currency_assets', Decimal('0'))
            stressed_data['foreign_currency_assets'] = fc_assets * (1 - depreciation)
            
            # Impact on foreign currency liabilities
            fc_liabilities = stressed_data.get('foreign_currency_liabilities', Decimal('0'))
            stressed_data['foreign_currency_liabilities'] = fc_liabilities * (1 + depreciation)
            
            # Impact on net income (through import/export businesses)
            net_income = stressed_data.get('net_income', Decimal('0'))
            stressed_data['net_income'] = net_income * (1 - scenario.get('export_income_decrease', Decimal('0')))
        
        # Liquidity Shock impacts
        if 'deposit_withdrawal' in scenario:
            withdrawal_rate = scenario['deposit_withdrawal']
            total_deposits = stressed_data.get('total_deposits', Decimal('0'))
            stressed_data['total_deposits'] = total_deposits * (1 - withdrawal_rate)
            
            # Impact on liquid assets
            liquid_assets = stressed_data.get('liquid_assets', Decimal('0'))
            decrease_rate = scenario.get('liquid_assets_decrease', Decimal('0'))
            stressed_data['liquid_assets'] = liquid_assets * (1 - decrease_rate)
            
            # Increase in funding costs
            interest_expense = stressed_data.get('interest_expense', Decimal('0'))
            funding_cost_increase = scenario.get('funding_cost_increase', Decimal('0'))
            stressed_data['interest_expense'] = interest_expense * (1 + funding_cost_increase)
        
        # Interest Rate Shock impacts
        if 'rate_increase' in scenario:
            rate_increase = scenario['rate_increase']
            
            # Impact on bond portfolio
            bond_portfolio = stressed_data.get('bond_portfolio', Decimal('0'))
            portfolio_loss = scenario.get('bond_portfolio_loss', Decimal('0'))
            stressed_data['bond_portfolio'] = bond_portfolio * (1 - portfolio_loss)
            
            # Impact on net interest margin
            net_interest_income = stressed_data.get('net_interest_income', Decimal('0'))
            nim_impact = scenario.get('net_interest_margin_impact', Decimal('0'))
            stressed_data['net_interest_income'] = net_interest_income * (1 - nim_impact)
        
        # Macroeconomic Downturn impacts
        if 'gdp_contraction' in scenario:
            gdp_contraction = scenario['gdp_contraction']
            
            # Increase in loan defaults
            default_increase = scenario.get('loan_default_increase', Decimal('0'))
            gross_npa = stressed_data.get('gross_npa', Decimal('0'))
            total_loans = stressed_data.get('total_loans', Decimal('1'))
            
            new_npa = gross_npa + (total_loans * default_increase)
            stressed_data['gross_npa'] = new_npa
            stressed_data['net_npa'] = new_npa  # Simplified
            
            # Impact on asset values
            asset_price_decline = scenario.get('asset_price_decline', Decimal('0'))
            total_assets = stressed_data.get('total_assets', Decimal('0'))
            stressed_data['total_assets'] = total_assets * (1 - asset_price_decline)
            
            # Impact on earnings
            net_income = stressed_data.get('net_income', Decimal('0'))
            stressed_data['net_income'] = net_income * (1 - gdp_contraction * 2)  # Amplified impact
        
        # Sectoral Shock impacts
        if 'sector_default_rate' in scenario:
            sector_default_rate = scenario['sector_default_rate']
            
            # Assume some portion of portfolio is in affected sector
            sector_exposure = stressed_data.get('sector_exposure', Decimal('0.3'))  # 30% default
            additional_defaults = total_loans * sector_exposure * sector_default_rate
            
            stressed_data['gross_npa'] = stressed_data.get('gross_npa', Decimal('0')) + additional_defaults
            
            # Collateral value decline
            collateral_value_decline = scenario.get('collateral_value_decline', Decimal('0'))
            collateral_cover = stressed_data.get('collateral_cover', Decimal('1'))
            stressed_data['collateral_cover'] = collateral_cover * (1 - collateral_value_decline)
        
        return stressed_data
    
    def _calculate_impact(self, current_data: Dict, stressed_data: Dict) -> Dict:
        """Calculate impact of stress scenario"""
        
        impact = {}
        
        # Key financial metrics to compare
        metrics = [
            'total_assets', 'total_equity', 'net_income', 'gross_npa', 
            'liquid_assets', 'capital_adequacy_ratio', 'return_on_assets'
        ]
        
        for metric in metrics:
            current_val = current_data.get(metric, Decimal('0'))
            stressed_val = stressed_data.get(metric, current_val)
            
            if current_val != 0:
                change_pct = ((stressed_val - current_val) / current_val) * 100
                impact[metric] = {
                    'current': float(current_val),
                    'stressed': float(stressed_val),
                    'change_percentage': float(change_pct),
                    'absolute_change': float(stressed_val - current_val)
                }
        
        # Calculate regulatory ratios impact
        impact['regulatory_ratios'] = self._calculate_regulatory_impact(current_data, stressed_data)
        
        return impact
    
    def _calculate_regulatory_impact(self, current_data: Dict, stressed_data: Dict) -> Dict:
        """Calculate impact on regulatory ratios"""
        
        # Calculate current ratios
        current_ratios = self._calculate_regulatory_ratios(current_data)
        stressed_ratios = self._calculate_regulatory_ratios(stressed_data)
        
        impact = {}
        for ratio, current_val in current_ratios.items():
            stressed_val = stressed_ratios.get(ratio, current_val)
            change_pct = ((stressed_val - current_val) / current_val) * 100 if current_val != 0 else 0
            
            impact[ratio] = {
                'current': float(current_val),
                'stressed': float(stressed_val),
                'change_percentage': float(change_pct),
                'meets_requirement': self._check_regulatory_requirement(ratio, stressed_val)
            }
        
        return impact
    
    def _calculate_regulatory_ratios(self, data: Dict) -> Dict:
        """Calculate key regulatory ratios"""
        
        ratios = {}
        
        # Capital Adequacy Ratio
        tier1_capital = data.get('tier1_capital', Decimal('0'))
        tier2_capital = data.get('tier2_capital', Decimal('0'))
        risk_weighted_assets = data.get('risk_weighted_assets', Decimal('1'))
        ratios['car'] = ((tier1_capital + tier2_capital) / risk_weighted_assets) * 100
        
        # Tier 1 Ratio
        ratios['tier1_ratio'] = (tier1_capital / risk_weighted_assets) * 100
        
        # Liquidity Coverage Ratio (simplified)
        liquid_assets = data.get('liquid_assets', Decimal('0'))
        total_assets = data.get('total_assets', Decimal('1'))
        ratios['liquidity_ratio'] = (liquid_assets / total_assets) * 100
        
        # NPA Ratio
        gross_npa = data.get('gross_npa', Decimal('0'))
        total_loans = data.get('total_loans', Decimal('1'))
        ratios['npa_ratio'] = (gross_npa / total_loans) * 100
        
        return ratios
    
    def _calculate_capital_impact(self, current_data: Dict, stressed_data: Dict) -> Dict:
        """Calculate impact on capital adequacy"""
        
        # Current capital
        current_tier1 = current_data.get('tier1_capital', Decimal('0'))
        current_tier2 = current_data.get('tier2_capital', Decimal('0'))
        current_rwa = current_data.get('risk_weighted_assets', Decimal('1'))
        current_car = ((current_tier1 + current_tier2) / current_rwa) * 100
        
        # Stressed capital
        stressed_tier1 = stressed_data.get('tier1_capital', current_tier1)
        stressed_tier2 = stressed_data.get('tier2_capital', current_tier2)
        stressed_rwa = stressed_data.get('risk_weighted_assets', current_rwa)
        stressed_car = ((stressed_tier1 + stressed_tier2) / stressed_rwa) * 100
        
        # Regulatory minimum
        regulatory_min = Decimal('10.0')  # 10% CAR requirement
        capital_shortfall = max(Decimal('0'), regulatory_min - stressed_car)
        
        # Additional capital needed
        additional_capital = (capital_shortfall * stressed_rwa) / 100 if stressed_car < regulatory_min else Decimal('0')
        
        return {
            'current_car': float(current_car),
            'stressed_car': float(stressed_car),
            'regulatory_minimum': float(regulatory_min),
            'capital_shortfall_percentage': float(capital_shortfall),
            'additional_capital_required': float(additional_capital),
            'meets_requirements': stressed_car >= regulatory_min
        }
    
    def _calculate_pd_under_stress(self, current_data: Dict, stressed_data: Dict) -> float:
        """Calculate Probability of Default under stress"""
        
        # Simplified PD model using financial ratios
        car = self._calculate_regulatory_ratios(stressed_data)['car']
        npa_ratio = self._calculate_regulatory_ratios(stressed_data)['npa_ratio']
        roa = (stressed_data.get('net_income', Decimal('0')) / stressed_data.get('total_assets', Decimal('1'))) * 100
        
        # Logistic regression-like approach (simplified)
        z_score = (
            car * 0.3 -           # Higher CAR reduces PD
            npa_ratio * 0.4 +     # Higher NPA increases PD
            roa * 0.2 -           # Higher ROA reduces PD
            5.0                   # Constant adjustment
        )
        
        # Convert to probability using logistic function
        pd_stressed = 1 / (1 + np.exp(-z_score))
        
        return float(min(pd_stressed, 0.99))  # Cap at 99%
    
    def _calculate_liquidity_gap(self, stressed_data: Dict) -> float:
        """Calculate liquidity gap under stress"""
        
        liquid_assets = stressed_data.get('liquid_assets', Decimal('0'))
        short_term_liabilities = stressed_data.get('short_term_liabilities', Decimal('1'))
        
        liquidity_gap = ((liquid_assets - short_term_liabilities) / short_term_liabilities) * 100
        
        return float(liquidity_gap)
    
    def _calculate_camels_deterioration(self, current_camels: Dict, stressed_camels: Dict) -> Dict:
        """Calculate deterioration in CAMELS rating"""
        
        deterioration = {}
        
        components = ['capital_adequacy', 'asset_quality', 'management_quality', 
                     'earnings', 'liquidity', 'sensitivity']
        
        for component in components:
            current_score = current_camels[component]['score']
            stressed_score = stressed_camels[component]['score']
            score_deterioration = stressed_score - current_score
            
            deterioration[component] = {
                'current_score': current_score,
                'stressed_score': stressed_score,
                'deterioration': score_deterioration,
                'rating_change': f"{current_camels[component]['rating']} -> {stressed_camels[component]['rating']}"
            }
        
        # Overall composite deterioration
        current_composite = current_camels['composite_rating']
        stressed_composite = stressed_camels['composite_rating']
        
        deterioration['composite'] = {
            'current_rating': current_composite,
            'stressed_rating': stressed_composite,
            'deterioration': stressed_composite - current_composite,
            'risk_grade_change': f"{current_camels['risk_grade']} -> {stressed_camels['risk_grade']}"
        }
        
        return deterioration
    
    def _check_regulatory_requirement(self, ratio: str, value: Decimal) -> bool:
        """Check if ratio meets regulatory requirement"""
        
        requirements = {
            'car': Decimal('10.0'),      # 10% minimum CAR
            'tier1_ratio': Decimal('6.0'), # 6% minimum Tier 1
            'liquidity_ratio': Decimal('20.0'), # 20% minimum liquidity
            'npa_ratio': Decimal('5.0')   # 5% maximum NPA
        }
        
        if ratio in ['car', 'tier1_ratio', 'liquidity_ratio']:
            return value >= requirements.get(ratio, Decimal('0'))
        elif ratio == 'npa_ratio':
            return value <= requirements.get(ratio, Decimal('100'))
        
        return True
    
    def _generate_recommendations(self, impact_analysis: Dict, stressed_camels: Dict) -> List[str]:
        """Generate recommendations based on stress test results"""
        
        recommendations = []
        
        # Capital adequacy recommendations
        capital_impact = impact_analysis.get('regulatory_ratios', {}).get('car', {})
        if not capital_impact.get('meets_requirement', True):
            recommendations.append(
                "Immediate capital injection required to meet regulatory capital adequacy requirements"
            )
        
        # Liquidity recommendations
        liquidity_impact = impact_analysis.get('regulatory_ratios', {}).get('liquidity_ratio', {})
        if not liquidity_impact.get('meets_requirement', True):
            recommendations.append(
                "Implement liquidity contingency plan and diversify funding sources"
            )
        
        # Asset quality recommendations
        if stressed_camels['asset_quality']['rating'] in ['MARGINAL', 'UNSATISFACTORY']:
            recommendations.append(
                "Enhance credit risk management and increase provisions for non-performing assets"
            )
        
        # Sensitivity recommendations
        if stressed_camels['sensitivity']['rating'] in ['MARGINAL', 'UNSATISFACTORY']:
            recommendations.append(
                "Implement hedging strategies to reduce foreign exchange and interest rate risk exposure"
            )
        
        # Overall risk management
        if stressed_camels['composite_rating'] >= 3.5:
            recommendations.append(
                "Develop comprehensive risk mitigation strategy and enhance stress testing frequency"
            )
        
        # Early warning indicators
        if any(comp['rating'] in ['MARGINAL', 'UNSATISFACTORY'] for comp in stressed_camels.values() 
               if isinstance(comp, dict) and 'rating' in comp):
            recommendations.append(
                "Activate early warning system and increase supervisory monitoring intensity"
            )
        
        return recommendations