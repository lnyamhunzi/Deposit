import numpy as np
from decimal import Decimal
from typing import Dict, Tuple, List
import json

class CAMELSEngine:
    
    def __init__(self):
        self.rating_thresholds = {
            'capital': {
                1: {'car': 15, 'tier1': 10},
                2: {'car': 12, 'tier1': 8},
                3: {'car': 10, 'tier1': 6},
                4: {'car': 8, 'tier1': 4},
                5: {'car': 0, 'tier1': 0}
            },
            'assets': {
                1: {'npl': 2, 'reserve': 150},
                2: {'npl': 4, 'reserve': 120},
                3: {'npl': 6, 'reserve': 100},
                4: {'npl': 10, 'reserve': 75},
                5: {'npl': float('inf'), 'reserve': 0}
            },
            'earnings': {
                1: {'roa': 1.5, 'roe': 15, 'nim': 4.0},
                2: {'roa': 1.0, 'roe': 12, 'nim': 3.5},
                3: {'roa': 0.5, 'roe': 8, 'nim': 3.0},
                4: {'roa': 0, 'roe': 5, 'nim': 2.0},
                5: {'roa': -float('inf'), 'roe': -float('inf'), 'nim': 0}
            },
            'liquidity': {
                1: {'ratio': 30, 'cash': 15, 'ltd': 80},
                2: {'ratio': 25, 'cash': 12, 'ltd': 90},
                3: {'ratio': 20, 'cash': 10, 'ltd': 100},
                4: {'ratio': 15, 'cash': 7, 'ltd': 110},
                5: {'ratio': 0, 'cash': 0, 'ltd': float('inf')}
            }
        }
    
    def calculate_capital_adequacy(self, financial_data: Dict) -> Tuple[int, float, Dict]:
        total_capital = float(financial_data.get('total_capital', 0))
        tier1_capital = float(financial_data.get('tier1_capital', 0))
        tier2_capital = float(financial_data.get('tier2_capital', 0))
        risk_weighted_assets = float(financial_data.get('risk_weighted_assets', 1))
        total_assets = float(financial_data.get('total_assets', 1))
        
        if risk_weighted_assets == 0:
            risk_weighted_assets = 1
        if total_assets == 0:
            total_assets = 1
        
        car = (total_capital / risk_weighted_assets) * 100
        tier1_ratio = (tier1_capital / risk_weighted_assets) * 100
        tier2_ratio = (tier2_capital / risk_weighted_assets) * 100
        capital_to_assets = (total_capital / total_assets) * 100
        
        rating = self._get_capital_rating(car, tier1_ratio)
        score = self._rating_to_score(rating)
        
        metrics = {
            'capital_adequacy_ratio': round(car, 4),
            'tier1_capital_ratio': round(tier1_ratio, 4),
            'tier2_capital_ratio': round(tier2_ratio, 4),
            'capital_to_assets_ratio': round(capital_to_assets, 4)
        }
        
        return rating, score, metrics
    
    def _get_capital_rating(self, car: float, tier1_ratio: float) -> int:
        if car >= 15 and tier1_ratio >= 10:
            return 1
        elif car >= 12 and tier1_ratio >= 8:
            return 2
        elif car >= 10 and tier1_ratio >= 6:
            return 3
        elif car >= 8 and tier1_ratio >= 4:
            return 4
        else:
            return 5
    
    def calculate_asset_quality(self, financial_data: Dict) -> Tuple[int, float, Dict]:
        non_performing_loans = float(financial_data.get('non_performing_loans', 0))
        total_loans = float(financial_data.get('total_loans', 1))
        loan_loss_reserves = float(financial_data.get('loan_loss_reserves', 0))
        total_assets = float(financial_data.get('total_assets', 1))
        
        if total_loans == 0:
            total_loans = 1
        if non_performing_loans == 0:
            non_performing_loans = 0.01
            
        npl_ratio = (non_performing_loans / total_loans) * 100
        reserve_ratio = (loan_loss_reserves / non_performing_loans) * 100 if non_performing_loans > 0 else 150
        concentration = self._calculate_concentration(financial_data)
        
        rating = self._get_asset_rating(npl_ratio, reserve_ratio)
        score = self._rating_to_score(rating)
        
        metrics = {
            'npl_ratio': round(npl_ratio, 4),
            'loan_loss_reserve_ratio': round(reserve_ratio, 4),
            'asset_concentration': round(concentration, 4)
        }
        
        return rating, score, metrics
    
    def _get_asset_rating(self, npl_ratio: float, reserve_ratio: float) -> int:
        if npl_ratio < 2 and reserve_ratio >= 150:
            return 1
        elif npl_ratio < 4 and reserve_ratio >= 120:
            return 2
        elif npl_ratio < 6 and reserve_ratio >= 100:
            return 3
        elif npl_ratio < 10 and reserve_ratio >= 75:
            return 4
        else:
            return 5
    
    def _calculate_concentration(self, financial_data: Dict) -> float:
        top_10_exposures = float(financial_data.get('top_10_exposures', 0))
        total_loans = float(financial_data.get('total_loans', 1))
        return (top_10_exposures / total_loans) * 100 if total_loans > 0 else 0
    
    def calculate_management_quality(self, financial_data: Dict, qualitative_data: Dict = None) -> Tuple[int, float, Dict]:
        if qualitative_data is None:
            qualitative_data = {}
        
        quality_score = float(qualitative_data.get('management_quality', 3.0))
        internal_controls = float(qualitative_data.get('internal_controls', 3.0))
        risk_mgmt = float(qualitative_data.get('risk_management', 3.0))
        board_oversight = float(qualitative_data.get('board_oversight', 3.0))
        
        avg_score = (quality_score + internal_controls + risk_mgmt + board_oversight) / 4
        
        rating = self._score_to_rating(avg_score)
        
        metrics = {
            'management_quality': round(quality_score, 2),
            'internal_controls': round(internal_controls, 2),
            'risk_management': round(risk_mgmt, 2),
            'board_oversight': round(board_oversight, 2)
        }
        
        return rating, round(avg_score, 2), metrics
    
    def calculate_earnings(self, financial_data: Dict) -> Tuple[int, float, Dict]:
        net_income = float(financial_data.get('net_income', 0))
        total_assets = float(financial_data.get('total_assets', 1))
        total_equity = float(financial_data.get('total_equity', 1))
        interest_income = float(financial_data.get('interest_income', 0))
        interest_expense = float(financial_data.get('interest_expense', 0))
        avg_earning_assets = float(financial_data.get('average_earning_assets', total_assets))
        
        if total_assets == 0:
            total_assets = 1
        if total_equity == 0:
            total_equity = 1
        if avg_earning_assets == 0:
            avg_earning_assets = 1
            
        roa = (net_income / total_assets) * 100
        roe = (net_income / total_equity) * 100
        nim = ((interest_income - interest_expense) / avg_earning_assets) * 100
        
        earnings_trend = float(financial_data.get('earnings_growth', 0))
        
        rating = self._get_earnings_rating(roa, roe, nim)
        score = self._rating_to_score(rating)
        
        metrics = {
            'return_on_assets': round(roa, 4),
            'return_on_equity': round(roe, 4),
            'net_interest_margin': round(nim, 4),
            'earnings_trend': round(earnings_trend, 4)
        }
        
        return rating, score, metrics
    
    def _get_earnings_rating(self, roa: float, roe: float, nim: float) -> int:
        if roa >= 1.5 and roe >= 15 and nim >= 4.0:
            return 1
        elif roa >= 1.0 and roe >= 12 and nim >= 3.5:
            return 2
        elif roa >= 0.5 and roe >= 8 and nim >= 3.0:
            return 3
        elif roa >= 0 and roe >= 5 and nim >= 2.0:
            return 4
        else:
            return 5
    
    def calculate_liquidity(self, financial_data: Dict) -> Tuple[int, float, Dict]:
        liquid_assets = float(financial_data.get('liquid_assets', 0))
        cash_equivalents = float(financial_data.get('cash_equivalents', 0))
        total_assets = float(financial_data.get('total_assets', 1))
        total_loans = float(financial_data.get('total_loans', 0))
        total_deposits = float(financial_data.get('total_deposits', 1))
        
        if total_assets == 0:
            total_assets = 1
        if total_deposits == 0:
            total_deposits = 1
            
        liquidity_ratio = (liquid_assets / total_assets) * 100
        cash_ratio = (cash_equivalents / total_assets) * 100
        liquid_assets_ratio = liquidity_ratio
        ltd_ratio = (total_loans / total_deposits) * 100
        
        rating = self._get_liquidity_rating(liquidity_ratio, cash_ratio, ltd_ratio)
        score = self._rating_to_score(rating)
        
        metrics = {
            'liquidity_ratio': round(liquidity_ratio, 4),
            'cash_to_assets_ratio': round(cash_ratio, 4),
            'liquid_assets_ratio': round(liquid_assets_ratio, 4),
            'loan_to_deposit_ratio': round(ltd_ratio, 4)
        }
        
        return rating, score, metrics
    
    def _get_liquidity_rating(self, liq_ratio: float, cash_ratio: float, ltd_ratio: float) -> int:
        if liq_ratio >= 30 and cash_ratio >= 15 and ltd_ratio <= 80:
            return 1
        elif liq_ratio >= 25 and cash_ratio >= 12 and ltd_ratio <= 90:
            return 2
        elif liq_ratio >= 20 and cash_ratio >= 10 and ltd_ratio <= 100:
            return 3
        elif liq_ratio >= 15 and cash_ratio >= 7 and ltd_ratio <= 110:
            return 4
        else:
            return 5
    
    def calculate_sensitivity(self, financial_data: Dict, scenario_data: Dict = None) -> Tuple[int, float, Dict]:
        if scenario_data is None:
            scenario_data = {}
        
        interest_rate_gap = float(financial_data.get('interest_rate_gap', 0))
        rate_sensitive_assets = float(financial_data.get('rate_sensitive_assets', 0))
        total_assets = float(financial_data.get('total_assets', 1))
        fx_exposure = float(financial_data.get('fx_exposure', 0))
        
        if total_assets == 0:
            total_assets = 1
            
        ir_risk = abs(interest_rate_gap / total_assets) * 100
        fx_risk = abs(fx_exposure / total_assets) * 100
        concentration = self._calculate_sector_concentration(financial_data)
        
        rating = self._get_sensitivity_rating(ir_risk, fx_risk, concentration)
        score = self._rating_to_score(rating)
        
        metrics = {
            'interest_rate_risk': round(ir_risk, 4),
            'fx_risk': round(fx_risk, 4),
            'market_concentration': round(concentration, 4)
        }
        
        return rating, score, metrics
    
    def _get_sensitivity_rating(self, ir_risk: float, fx_risk: float, concentration: float) -> int:
        total_risk = ir_risk + fx_risk + (concentration / 10)
        
        if total_risk < 10:
            return 1
        elif total_risk < 20:
            return 2
        elif total_risk < 30:
            return 3
        elif total_risk < 40:
            return 4
        else:
            return 5
    
    def _calculate_sector_concentration(self, financial_data: Dict) -> float:
        sector_exposures = financial_data.get('sector_exposures', {})
        if not sector_exposures:
            return 20.0
        
        total_exposure = sum(sector_exposures.values())
        if total_exposure == 0:
            return 0
        
        max_exposure = max(sector_exposures.values())
        return (max_exposure / total_exposure) * 100
    
    def calculate_composite_rating(self, individual_ratings: Dict[str, int]) -> Tuple[int, float]:
        weights = {
            'capital': 0.20,
            'assets': 0.20,
            'management': 0.25,
            'earnings': 0.15,
            'liquidity': 0.10,
            'sensitivity': 0.10
        }
        
        weighted_sum = 0
        for component, rating in individual_ratings.items():
            weight = weights.get(component, 0)
            weighted_sum += rating * weight
        
        composite_rating = round(weighted_sum)
        composite_rating = max(1, min(5, composite_rating))
        
        composite_score = self._rating_to_score(composite_rating)
        
        return composite_rating, composite_score
    
    def _rating_to_score(self, rating: int) -> float:
        score_map = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0}
        return score_map.get(rating, 3.0)
    
    def _score_to_rating(self, score: float) -> int:
        if score >= 4.5:
            return 1
        elif score >= 3.5:
            return 2
        elif score >= 2.5:
            return 3
        elif score >= 1.5:
            return 4
        else:
            return 5
    
    def generate_early_warning_indicators(self, financial_data: Dict, camels_ratings: Dict) -> List[Dict]:
        warnings = []
        
        if camels_ratings.get('capital') >= 4:
            warnings.append({
                'type': 'Capital Deficiency',
                'severity': 'High',
                'message': 'Capital adequacy ratio below minimum requirements'
            })
        
        if camels_ratings.get('assets') >= 4:
            warnings.append({
                'type': 'Asset Quality',
                'severity': 'High',
                'message': 'High non-performing loan ratio detected'
            })
        
        if camels_ratings.get('earnings') >= 4:
            warnings.append({
                'type': 'Earnings Weakness',
                'severity': 'Medium',
                'message': 'Declining profitability and earnings trend'
            })
        
        if camels_ratings.get('liquidity') >= 4:
            warnings.append({
                'type': 'Liquidity Stress',
                'severity': 'High',
                'message': 'Insufficient liquid assets to meet obligations'
            })
        
        composite = camels_ratings.get('composite', 3)
        if composite >= 4:
            warnings.append({
                'type': 'Overall Financial Health',
                'severity': 'Critical',
                'message': 'Composite CAMELS rating indicates significant distress'
            })
        
        return warnings
    
    def determine_risk_level(self, composite_rating: int) -> str:
        risk_levels = {
            1: 'Low Risk',
            2: 'Moderate Risk',
            3: 'Medium Risk',
            4: 'High Risk',
            5: 'Critical Risk'
        }
        return risk_levels.get(composite_rating, 'Medium Risk')
    
    def generate_recommendations(self, camels_ratings: Dict, early_warnings: List[Dict]) -> str:
        recommendations = []
        
        if camels_ratings.get('capital', 3) >= 3:
            recommendations.append("• Increase capital base through retained earnings or capital injection")
            recommendations.append("• Review dividend policy to preserve capital")
        
        if camels_ratings.get('assets', 3) >= 3:
            recommendations.append("• Strengthen loan underwriting standards")
            recommendations.append("• Increase provisioning for non-performing loans")
            recommendations.append("• Implement aggressive recovery strategies")
        
        if camels_ratings.get('management', 3) >= 3:
            recommendations.append("• Enhance risk management framework")
            recommendations.append("• Strengthen internal controls and audit functions")
            recommendations.append("• Improve board oversight and governance")
        
        if camels_ratings.get('earnings', 3) >= 3:
            recommendations.append("• Diversify revenue streams")
            recommendations.append("• Improve cost efficiency and reduce operating expenses")
            recommendations.append("• Optimize interest rate spreads")
        
        if camels_ratings.get('liquidity', 3) >= 3:
            recommendations.append("• Build liquid asset buffers")
            recommendations.append("• Diversify funding sources")
            recommendations.append("• Implement robust liquidity contingency plan")
        
        if camels_ratings.get('sensitivity', 3) >= 3:
            recommendations.append("• Implement hedging strategies for interest rate and FX risks")
            recommendations.append("• Reduce sector and geographic concentration")
            recommendations.append("• Enhance asset-liability management")
        
        if len(early_warnings) > 2:
            recommendations.append("• Immediate supervisory intervention recommended")
            recommendations.append("• Develop comprehensive recovery plan")
        
    def run_stress_test(self, financial_data: Dict, scenario: Dict) -> Dict:
        stressed_data = financial_data.copy()
        impacts = {}

        if scenario.get('interest_rate_shock'):
            shock = scenario['interest_rate_shock']
            stressed_data['net_income'] *= (1 - shock / 100)
            impacts['earnings'] = f'Net income reduced by {shock}%'

        if scenario.get('npl_increase'):
            increase = scenario['npl_increase']
            stressed_data['non_performing_loans'] *= (1 + increase / 100)
            impacts['assets'] = f'NPLs increased by {increase}%'

        if scenario.get('liquidity_crisis'):
            reduction = scenario['liquidity_crisis']
            stressed_data['liquid_assets'] *= (1 - reduction / 100)
            impacts['liquidity'] = f'Liquid assets reduced by {reduction}%'

        # Recalculate CAMELS with stressed data
        c_rating, _, _ = self.calculate_capital_adequacy(stressed_data)
        a_rating, _, _ = self.calculate_asset_quality(stressed_data)
        m_rating, _, _ = self.calculate_management_quality(stressed_data)
        e_rating, _, _ = self.calculate_earnings(stressed_data)
        l_rating, _, _ = self.calculate_liquidity(stressed_data)
        s_rating, _, _ = self.calculate_sensitivity(stressed_data)

        stressed_ratings = {
            'capital': c_rating,
            'assets': a_rating,
            'management': m_rating,
            'earnings': e_rating,
            'liquidity': l_rating,
            'sensitivity': s_rating
        }

        stressed_composite, _ = self.calculate_composite_rating(stressed_ratings)

        return {
            'scenario_name': scenario['name'],
            'impacts': impacts,
            'stressed_composite_rating': stressed_composite,
            'stressed_ratings': stressed_ratings
        }
