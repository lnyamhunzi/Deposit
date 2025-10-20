import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, List, Tuple
import json
from decimal import Decimal
import os

MODEL_DIR = 'ml_models'

class RiskAnalysisEngine:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.failure_model = None
        self.pd_model = None
        self.anomaly_detector = None
        self.feature_names = [
            'capital_adequacy_ratio', 'npl_ratio', 'roa', 'roe',
            'liquidity_ratio', 'loan_to_deposit_ratio', 'cost_to_income',
            'deposit_growth', 'loan_growth', 'equity_to_assets'
        ]
        
        # Try to load trained models
        self._load_trained_models()
    
    def calculate_pd_lgd_ead(self, financial_data: Dict, camels_rating: int) -> Dict:
        capital_ratio = float(financial_data.get('capital_adequacy_ratio', 10))
        npl_ratio = float(financial_data.get('npl_ratio', 5))
        roa = float(financial_data.get('return_on_assets', 1))
        liquidity_ratio = float(financial_data.get('liquidity_ratio', 20))
        
        # Use ML model for PD if available, otherwise fall back to heuristics
        if self.pd_model is not None:
            # Extract features for ML prediction
            features = []
            for feature in self.feature_names:
                value = float(financial_data.get(feature, 0))
                features.append(value)
            
            try:
                features_scaled = self.scaler.transform([features])
                
                # Get ML-based PD probability
                ml_pd_prob = self.pd_model.predict_proba(features_scaled)[0][1]
                
                # Convert probability to actual PD value
                # High probability (>0.5) means high risk, map to higher PD values
                if ml_pd_prob > 0.8:
                    base_pd = 0.15 + (ml_pd_prob - 0.8) * 0.25  # 0.15 to 0.20
                elif ml_pd_prob > 0.6:
                    base_pd = 0.05 + (ml_pd_prob - 0.6) * 0.5  # 0.05 to 0.15
                elif ml_pd_prob > 0.4:
                    base_pd = 0.01 + (ml_pd_prob - 0.4) * 0.2  # 0.01 to 0.05
                elif ml_pd_prob > 0.2:
                    base_pd = 0.005 + (ml_pd_prob - 0.2) * 0.025  # 0.005 to 0.01
                else:
                    base_pd = 0.001 + ml_pd_prob * 0.02  # 0.001 to 0.005
                
                # Blend with CAMELS-based adjustment (70% ML, 30% CAMELS)
                heuristic_pd = self._calculate_base_pd(capital_ratio, npl_ratio, roa, camels_rating)
                base_pd = base_pd * 0.7 + heuristic_pd * 0.3
                
                print(f"PD calculated using ML model: {base_pd:.6f} (ML prob: {ml_pd_prob:.4f}, Heuristic: {heuristic_pd:.6f})")
                
            except Exception as e:
                print(f"Error using ML model for PD, falling back to heuristics: {e}")
                base_pd = self._calculate_base_pd(capital_ratio, npl_ratio, roa, camels_rating)
        else:
            # Fall back to heuristic calculation
            base_pd = self._calculate_base_pd(capital_ratio, npl_ratio, roa, camels_rating)
            print(f"PD calculated using heuristics (no ML model): {base_pd:.6f}")
        
        lgd = self._calculate_lgd(npl_ratio, financial_data)
        
        total_exposure = float(financial_data.get('total_assets', 0))
        ead = total_exposure * 0.75
        
        expected_loss = base_pd * lgd * ead
        
        return {
            'probability_of_default': round(base_pd, 6),
            'loss_given_default': round(lgd, 6),
            'exposure_at_default': round(ead, 2),
            'expected_loss': round(expected_loss, 2)
        }
    
    def _calculate_base_pd(self, capital_ratio: float, npl_ratio: float, roa: float, camels_rating: int) -> float:
        rating_pd_map = {
            1: 0.0010,
            2: 0.0050,
            3: 0.0200,
            4: 0.0800,
            5: 0.2000
        }
        
        base_pd = rating_pd_map.get(camels_rating, 0.05)
        
        if capital_ratio < 8:
            base_pd *= 1.5
        elif capital_ratio < 10:
            base_pd *= 1.2
        
        if npl_ratio > 10:
            base_pd *= 1.8
        elif npl_ratio > 5:
            base_pd *= 1.3
        
        if roa < 0:
            base_pd *= 2.0
        elif roa < 0.5:
            base_pd *= 1.4
        
        return min(base_pd, 0.99)
    
    def _calculate_lgd(self, npl_ratio: float, financial_data: Dict) -> float:
        base_lgd = 0.45
        
        recovery_rate = float(financial_data.get('recovery_rate', 0.40))
        lgd = 1 - recovery_rate
        
        if npl_ratio > 15:
            lgd = min(lgd * 1.3, 0.85)
        elif npl_ratio > 10:
            lgd = min(lgd * 1.15, 0.75)
        
        collateral_quality = float(financial_data.get('collateral_quality_score', 0.6))
        lgd = lgd * (1 + (1 - collateral_quality) * 0.2)
        
        return min(max(lgd, 0.10), 0.95)
    
    def calculate_risk_scores(self, financial_data: Dict, camels_data: Dict) -> Dict:
        credit_risk = self._calculate_credit_risk(financial_data)
        market_risk = self._calculate_market_risk(financial_data)
        operational_risk = self._calculate_operational_risk(financial_data, camels_data)
        liquidity_risk = self._calculate_liquidity_risk(financial_data)
        
        weights = {
            'credit': 0.40,
            'market': 0.25,
            'operational': 0.20,
            'liquidity': 0.15
        }
        
        overall_risk = (
            credit_risk * weights['credit'] +
            market_risk * weights['market'] +
            operational_risk * weights['operational'] +
            liquidity_risk * weights['liquidity']
        )
        
        return {
            'overall_risk_score': round(overall_risk, 4),
            'credit_risk_score': round(credit_risk, 4),
            'market_risk_score': round(market_risk, 4),
            'operational_risk_score': round(operational_risk, 4),
            'liquidity_risk_score': round(liquidity_risk, 4)
        }
    
    def _calculate_credit_risk(self, financial_data: Dict) -> float:
        npl_ratio = float(financial_data.get('npl_ratio', 5))
        loan_concentration = float(financial_data.get('loan_concentration', 20))
        
        score = (npl_ratio * 5) + (loan_concentration / 5)
        return min(max(score, 0), 100)
    
    def _calculate_market_risk(self, financial_data: Dict) -> float:
        ir_sensitivity = float(financial_data.get('interest_rate_risk', 10))
        fx_exposure = float(financial_data.get('fx_risk', 5))
        
        score = (ir_sensitivity * 3) + (fx_exposure * 4)
        return min(max(score, 0), 100)
    
    def _calculate_operational_risk(self, financial_data: Dict, camels_data: Dict) -> float:
        management_score = float(camels_data.get('management_score', 3.0))
        
        rating = 6 - management_score
        score = rating * 20
        
        fraud_incidents = int(financial_data.get('fraud_incidents', 0))
        score += fraud_incidents * 5
        
        return min(max(score, 0), 100)
    
    def _calculate_liquidity_risk(self, financial_data: Dict) -> float:
        liquidity_ratio = float(financial_data.get('liquidity_ratio', 20))
        ltd_ratio = float(financial_data.get('loan_to_deposit_ratio', 85))
        
        liq_score = max(0, 50 - liquidity_ratio)
        ltd_score = max(0, (ltd_ratio - 80) / 2)
        
        score = liq_score + ltd_score
        return min(max(score, 0), 100)
    
    def train_failure_prediction_model(self, historical_data: pd.DataFrame) -> Dict:
        if len(historical_data) < 10:
            return {'status': 'insufficient_data', 'message': 'Need at least 10 historical records'}
        
        features = historical_data[self.feature_names].fillna(0)
        labels = historical_data['failed'].fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels if len(labels.unique()) > 1 else None
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.failure_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.failure_model.fit(X_train_scaled, y_train)
        
        train_score = self.failure_model.score(X_train_scaled, y_train)
        test_score = self.failure_model.score(X_test_scaled, y_test)
        
        feature_importance = dict(zip(self.feature_names, self.failure_model.feature_importances_))
        
        return {
            'status': 'success',
            'train_accuracy': round(train_score, 4),
            'test_accuracy': round(test_score, 4),
            'feature_importance': {k: round(v, 4) for k, v in feature_importance.items()}
        }
    
    def _load_trained_models(self):
        """Load pre-trained ML models from disk"""
        try:
            if os.path.exists(os.path.join(MODEL_DIR, 'scaler.pkl')):
                self.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
                self.failure_model = joblib.load(os.path.join(MODEL_DIR, 'failure_model.pkl'))
                self.pd_model = joblib.load(os.path.join(MODEL_DIR, 'pd_model.pkl'))
                self.anomaly_detector = joblib.load(os.path.join(MODEL_DIR, 'anomaly_detector.pkl'))
                print("ML models loaded successfully")
            else:
                print("No trained models found. Run ml_training.py to train models.")
        except Exception as e:
            print(f"Error loading ML models: {e}")
    
    def predict_bank_failure(self, financial_data: Dict) -> Dict:
        if self.failure_model is None:
            # If no model is loaded, return a warning prediction based on heuristics
            print("WARNING: No trained failure model available, using heuristics")
            
            car = float(financial_data.get('capital_adequacy_ratio', 12))
            npl = float(financial_data.get('npl_ratio', 3))
            roa = float(financial_data.get('roa', 1))
            
            # Simple heuristic-based probability
            failure_probability = 0.01
            if car < 8 or npl > 15 or roa < 0:
                failure_probability = 0.30
            elif car < 10 or npl > 10 or roa < 0.5:
                failure_probability = 0.15
            elif car < 12 or npl > 7 or roa < 1:
                failure_probability = 0.05
            
            risk_factors = []
            if car < 10:
                risk_factors.append('Low capital adequacy')
            if npl > 7:
                risk_factors.append('High non-performing loans')
            if roa < 0.5:
                risk_factors.append('Weak profitability')
            if financial_data.get('liquidity_ratio', 25) < 15:
                risk_factors.append('Liquidity stress')
            
            return {
                'ml_failure_probability': round(failure_probability, 6),
                'prediction': 'High Risk' if failure_probability > 0.2 else 'Normal',
                'risk_factors': risk_factors,
                'confidence': round(1 - failure_probability, 4),
                'model_status': 'heuristic_fallback'
            }
        
        # Use trained ML model
        features = []
        for feature in self.feature_names:
            value = float(financial_data.get(feature, 0))
            features.append(value)
        
        features_scaled = self.scaler.transform([features])
        
        failure_probability = self.failure_model.predict_proba(features_scaled)[0][1]
        prediction = self.failure_model.predict(features_scaled)[0]
        
        # Enhanced PD prediction using Logistic Regression model
        if self.pd_model is not None:
            pd_probability = self.pd_model.predict_proba(features_scaled)[0][1]
            # Weight both models
            failure_probability = (failure_probability * 0.7 + pd_probability * 0.3)
        
        risk_factors = []
        if financial_data.get('capital_adequacy_ratio', 12) < 10:
            risk_factors.append('Low capital adequacy')
        if financial_data.get('npl_ratio', 3) > 7:
            risk_factors.append('High non-performing loans')
        if financial_data.get('roa', 1) < 0.5:
            risk_factors.append('Weak profitability')
        if financial_data.get('liquidity_ratio', 25) < 15:
            risk_factors.append('Liquidity stress')
        
        return {
            'ml_failure_probability': round(failure_probability, 6),
            'prediction': 'High Risk' if prediction == 1 else 'Normal',
            'risk_factors': risk_factors,
            'confidence': round(max(failure_probability, 1 - failure_probability), 4),
            'model_status': 'ml_trained'
        }
    
    def detect_anomalies(self, current_data: Dict, historical_data: pd.DataFrame = None) -> Dict:
        if self.anomaly_detector is None:
            print("WARNING: No trained anomaly detector available")
            # Fallback to heuristic anomaly detection
            anomalies = []
            is_anomaly = False
            
            car = current_data.get('capital_adequacy_ratio', 12)
            npl = current_data.get('npl_ratio', 3)
            loan_growth = current_data.get('loan_growth', 5)
            
            if car < 8 or car > 25:
                anomalies.append({'metric': 'Capital Adequacy Ratio', 'value': car, 'reason': 'Unusual capital level'})
                is_anomaly = True
            
            if npl > 10:
                anomalies.append({'metric': 'NPL Ratio', 'value': npl, 'reason': 'Exceptionally high NPLs'})
                is_anomaly = True
            
            if loan_growth > 50 or loan_growth < -20:
                anomalies.append({'metric': 'Loan Growth', 'value': loan_growth, 'reason': 'Abnormal loan growth pattern'})
                is_anomaly = True
            
            return {
                'anomaly_score': -0.5 if is_anomaly else 0.5,
                'is_anomaly': is_anomaly,
                'anomalies_detected': anomalies,
                'model_status': 'heuristic_fallback'
            }
        
        # Use trained ML model
        features = [float(current_data.get(f, 0)) for f in self.feature_names]
        features_array = np.array([features])
        
        prediction = self.anomaly_detector.predict(features_array)[0]
        anomaly_score = self.anomaly_detector.score_samples(features_array)[0]
        
        is_anomaly = prediction == -1
        
        anomalies = []
        if is_anomaly:
            if current_data.get('capital_adequacy_ratio', 12) < 8 or current_data.get('capital_adequacy_ratio', 12) > 25:
                anomalies.append({'metric': 'Capital Adequacy Ratio', 'value': current_data.get('capital_adequacy_ratio'), 'reason': 'Unusual capital level'})
            
            if current_data.get('npl_ratio', 3) > 10:
                anomalies.append({'metric': 'NPL Ratio', 'value': current_data.get('npl_ratio'), 'reason': 'Exceptionally high NPLs'})
            
            if current_data.get('loan_growth', 5) > 50 or current_data.get('loan_growth', 5) < -20:
                anomalies.append({'metric': 'Loan Growth', 'value': current_data.get('loan_growth'), 'reason': 'Abnormal loan growth pattern'})
        
        return {
            'anomaly_score': round(float(anomaly_score), 4),
            'is_anomaly': is_anomaly,
            'anomalies_detected': anomalies,
            'model_status': 'ml_trained'
        }
    
    def run_stress_test(self, financial_data: Dict, scenarios: List[Dict]) -> Dict:
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'Unnamed Scenario')
            stress_factors = scenario.get('factors', {})
            
            stressed_data = financial_data.copy()
            
            if 'interest_rate_shock' in stress_factors:
                rate_shock = stress_factors['interest_rate_shock']
                nim = float(stressed_data.get('net_interest_margin', 3.5))
                stressed_data['net_interest_margin'] = nim * (1 + rate_shock / 100)
                
                ni = float(stressed_data.get('net_income', 1000000))
                stressed_data['net_income'] = ni * (1 + rate_shock / 200)
            
            if 'fx_shock' in stress_factors:
                fx_shock = stress_factors['fx_shock']
                fx_losses = float(stressed_data.get('total_assets', 0)) * abs(fx_shock) / 100 * 0.2
                
                ni = float(stressed_data.get('net_income', 1000000))
                stressed_data['net_income'] = ni - fx_losses
            
            if 'npl_increase' in stress_factors:
                npl_increase = stress_factors['npl_increase']
                npl = float(stressed_data.get('npl_ratio', 3))
                stressed_data['npl_ratio'] = npl * (1 + npl_increase / 100)
            
            if 'deposit_withdrawal' in stress_factors:
                withdrawal = stress_factors['deposit_withdrawal']
                deposits = float(stressed_data.get('total_deposits', 0))
                stressed_data['total_deposits'] = deposits * (1 - withdrawal / 100)
                
                liquid_assets = float(stressed_data.get('liquid_assets', 0))
                stressed_data['liquid_assets'] = liquid_assets - (deposits * withdrawal / 100)
            
            total_assets = float(stressed_data.get('total_assets', 1))
            total_equity = float(stressed_data.get('total_equity', 1))
            
            stressed_roa = (float(stressed_data.get('net_income', 0)) / total_assets) * 100
            stressed_roe = (float(stressed_data.get('net_income', 0)) / total_equity) * 100
            stressed_car = float(stressed_data.get('capital_adequacy_ratio', 12)) - (stressed_data.get('npl_ratio', 3) - financial_data.get('npl_ratio', 3)) / 2
            
            impact_assessment = self._assess_stress_impact(stressed_car, stressed_data.get('npl_ratio', 3), stressed_roa)
            
            results[scenario_name] = {
                'stressed_capital_ratio': round(stressed_car, 2),
                'stressed_npl_ratio': round(float(stressed_data.get('npl_ratio', 3)), 2),
                'stressed_roa': round(stressed_roa, 2),
                'stressed_roe': round(stressed_roe, 2),
                'stressed_liquidity': round((float(stressed_data.get('liquid_assets', 0)) / total_assets) * 100, 2),
                'impact': impact_assessment,
                'passes_stress_test': stressed_car > 8 and stressed_roa > -2
            }
        
        return results
    
    def _assess_stress_impact(self, car: float, npl: float, roa: float) -> str:
        if car < 8 or roa < -2 or npl > 15:
            return 'Severe - Bank would fail under this scenario'
        elif car < 10 or roa < 0 or npl > 10:
            return 'High - Significant capital erosion, intervention needed'
        elif car < 12 or roa < 0.5 or npl > 7:
            return 'Moderate - Manageable with corrective actions'
        else:
            return 'Low - Bank remains resilient'
    
    def determine_risk_category(self, overall_risk_score: float) -> str:
        if overall_risk_score < 20:
            return 'Low Risk'
        elif overall_risk_score < 40:
            return 'Moderate Risk'
        elif overall_risk_score < 60:
            return 'Medium Risk'
        elif overall_risk_score < 80:
            return 'High Risk'
        else:
            return 'Critical Risk'
    
    def determine_alert_level(self, risk_category: str, failure_probability: float, anomaly_detected: bool) -> str:
        if risk_category in ['Critical Risk', 'High Risk'] or failure_probability > 0.3:
            return 'Red'
        elif risk_category == 'Medium Risk' or failure_probability > 0.1 or anomaly_detected:
            return 'Amber'
        elif risk_category == 'Moderate Risk' or failure_probability > 0.05:
            return 'Yellow'
        else:
            return 'Green'
