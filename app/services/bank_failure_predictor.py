import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy.orm import Session
from decimal import Decimal
from typing import Dict, List, Any
from datetime import datetime

class BankFailurePredictor:
    def __init__(self, db: Session):
        self.db = db
        self.models = {}
        self.scaler = None # Will be loaded
        self.feature_selector = None # Will be loaded
        self.best_model = None # Will be loaded
        self.model_path = "ml_models/failure_model.pkl"
        self.scaler_path = "ml_models/scaler.pkl"
        self.feature_selector_path = "ml_models/feature_selector.pkl" # Assuming a separate file for feature selector
        self._load_models()

    def _load_models(self):
        """Load pre-trained ML models and scaler"""
        try:
            if os.path.exists(self.model_path):
                self.best_model = joblib.load(self.model_path)
                print(f"Loaded best model from {self.model_path}")
            else:
                print(f"No pre-trained model found at {self.model_path}. Model will need to be trained.")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"Loaded scaler from {self.scaler_path}")
            else:
                print(f"No pre-trained scaler found at {self.scaler_path}. Scaler will need to be fitted.")

            if os.path.exists(self.feature_selector_path):
                self.feature_selector = joblib.load(self.feature_selector_path)
                print(f"Loaded feature selector from {self.feature_selector_path}")
            else:
                print(f"No pre-trained feature selector found at {self.feature_selector_path}. Feature selector will need to be fitted.")

        except Exception as e:
            print(f"Error loading pre-trained models: {e}")
            self.best_model = None
            self.scaler = None
            self.feature_selector = None
    
    def train_models(self, historical_data: pd.DataFrame, target_column: str = 'failed') -> Dict:
        """Train multiple ML models for bank failure prediction"""
        
        # Prepare data
        X, y, feature_names = self._prepare_training_data(historical_data, target_column)
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=15)
        X_selected = self.feature_selector.fit_transform(X_resampled, y_resampled)
        
        # Get selected feature names
        selected_features = feature_names[self.feature_selector.get_support()]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
        
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                auc_score = roc_auc_score(y_test, y_pred_proba)
                cv_scores = cross_val_score(model, X_scaled, y_resampled, cv=5, scoring='roc_auc')
                
                results[name] = {
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': self._get_feature_importance(model, selected_features),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                print(f"{name} - AUC: {auc_score:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Select best model
        best_model_name = max(results.keys(), 
                            key=lambda x: results[x].get('auc_score', 0) if 'auc_score' in results[x] else 0)
        
        self.best_model = self.models[best_model_name]
        self.feature_importance = results[best_model_name]['feature_importance']

        # Save the trained model, scaler, and feature selector
        joblib.dump(self.best_model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.feature_selector, self.feature_selector_path)
        print(f"Trained models, scaler, and feature selector saved to {self.model_path}, {self.scaler_path}, {self.feature_selector_path}")
        
        return {
            'best_model': best_model_name,
            'results': results,
            'selected_features': selected_features.tolist(),
            'training_summary': {
                'total_samples': len(X),
                'failure_cases': sum(y),
                'non_failure_cases': len(y) - sum(y),
                'selected_features_count': len(selected_features)
            }
        }
    
    def predict_failure_risk(self, institution_data: Dict) -> Dict:
        """Predict bank failure risk for a single institution"""
        
        try:
            if self.feature_selector is None or self.best_model is None or self.scaler is None:
                return {"error": "Pre-trained models, scaler, or feature selector not loaded."}
            
            # Prepare features as a dictionary
            features_dict = self._prepare_prediction_features_dict(institution_data)
            features_df = pd.DataFrame([features_dict])

            # Ensure columns are in the same order as training data
            # This requires storing the original feature names from training
            # For now, assume features_df columns match expected features for feature_selector
            
            # Select features
            features_selected = self.feature_selector.transform(features_df)
            
            # Scale features
            features_scaled = self.scaler.transform(features_selected)
            
            # Predict
            failure_probability = self.best_model.predict_proba(features_scaled)[0, 1]
            
            # Risk categories
            risk_category = self._categorize_risk(failure_probability)
            
            # Key risk drivers
            risk_drivers = self._identify_risk_drivers(institution_data)
            
            return {
                'failure_probability': float(failure_probability),
                'risk_category': risk_category,
                'risk_level': self._get_risk_level(failure_probability),
                'early_warning_signals': self._detect_early_warnings(institution_data),
                'key_risk_drivers': risk_drivers,
                'prediction_confidence': self._calculate_confidence(failure_probability),
                'recommended_actions': self._generate_recommendations(risk_category, risk_drivers),
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _prepare_training_data(self, data: pd.DataFrame, target_column: str) -> tuple:
        """Prepare training data"""
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Convert categorical variables if any
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
        
        feature_names = X.columns
        
        return X.values, y.values, feature_names
    
    def _prepare_prediction_features_dict(self, institution_data: Dict) -> Dict:
        """Prepare features for prediction as a dictionary"""
        
        features = {
            'capital_adequacy_ratio': institution_data.get('capital_adequacy_ratio', 0),
            'tier1_ratio': institution_data.get('tier1_ratio', 0),
            'npa_ratio': institution_data.get('npa_ratio', 0),
            'provision_coverage_ratio': institution_data.get('provision_coverage_ratio', 0),
            'return_on_assets': institution_data.get('return_on_assets', 0),
            'return_on_equity': institution_data.get('return_on_equity', 0),
            'net_interest_margin': institution_data.get('net_interest_margin', 0),
            'cost_to_income_ratio': institution_data.get('cost_to_income_ratio', 0),
            'liquidity_ratio': institution_data.get('liquidity_ratio', 0),
            'loan_to_deposit_ratio': institution_data.get('loan_to_deposit_ratio', 0),
            'asset_growth_rate': institution_data.get('asset_growth_rate', 0),
            'deposit_growth_rate': institution_data.get('deposit_growth_rate', 0),
            'loan_growth_rate': institution_data.get('loan_growth_rate', 0),
            'operating_expense_ratio': institution_data.get('operating_expense_ratio', 0),
            'equity_to_assets_ratio': institution_data.get('equity_to_assets_ratio', 0),
            'earning_assets_ratio': institution_data.get('earning_assets_ratio', 0),
            'volatility_of_earnings': institution_data.get('volatility_of_earnings', 0),
            'concentration_risk': institution_data.get('concentration_risk', 0),
            'fx_exposure': institution_data.get('fx_exposure', 0),
            'interest_rate_gap': institution_data.get('interest_rate_gap', 0)
        }
        
        return features
    
    def _get_feature_importance(self, model, feature_names) -> Dict:
        """Get feature importance from model"""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        else:
            return {"message": "Feature importance not available for this model"}
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize failure risk"""
        if probability < 0.05:
            return "VERY_LOW"
        elif probability < 0.15:
            return "LOW"
        elif probability < 0.30:
            return "MODERATE"
        elif probability < 0.50:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _get_risk_level(self, probability: float) -> int:
        """Get risk level (1-5 scale)"""
        if probability < 0.05: return 1
        elif probability < 0.15: return 2
        elif probability < 0.30: return 3
        elif probability < 0.50: return 4
        else: return 5
    
    def _identify_risk_drivers(self, institution_data: Dict) -> List[Dict]:
        """Identify key risk drivers"""
        
        risk_drivers = []
        
        # Capital adequacy risk
        car = institution_data.get('capital_adequacy_ratio', 0)
        if car < 10:
            risk_drivers.append({
                'factor': 'Capital Adequacy',
                'value': car,
                'threshold': 10,
                'severity': 'HIGH' if car < 8 else 'MEDIUM'
            })
        
        # Asset quality risk
        npa_ratio = institution_data.get('npa_ratio', 0)
        if npa_ratio > 5:
            risk_drivers.append({
                'factor': 'NPA Ratio',
                'value': npa_ratio,
                'threshold': 5,
                'severity': 'HIGH' if npa_ratio > 10 else 'MEDIUM'
            })
        
        # Earnings risk
        roa = institution_data.get('return_on_assets', 0)
        if roa < 0.5:
            risk_drivers.append({
                'factor': 'Return on Assets',
                'value': roa,
                'threshold': 0.5,
                'severity': 'HIGH' if roa < 0 else 'MEDIUM'
            })
        
        # Liquidity risk
        liquidity_ratio = institution_data.get('liquidity_ratio', 0)
        if liquidity_ratio < 20:
            risk_drivers.append({
                'factor': 'Liquidity Ratio',
                'value': liquidity_ratio,
                'threshold': 20,
                'severity': 'HIGH' if liquidity_ratio < 15 else 'MEDIUM'
            })
        
        return sorted(risk_drivers, key=lambda x: x['severity'], reverse=True)[:5]
    
    def _detect_early_warnings(self, institution_data: Dict) -> List[str]:
        """Detect early warning signals"""
        
        warnings = []
        
        # Rapid growth warning
        asset_growth = institution_data.get('asset_growth_rate', 0)
        if asset_growth > 20:
            warnings.append(f"Rapid asset growth: {asset_growth:.1f}%")
        
        # Deteriorating asset quality
        npa_growth = institution_data.get('npa_growth_rate', 0)
        if npa_growth > 15:
            warnings.append(f"Rising NPA growth: {npa_growth:.1f}%")
        
        # Declining profitability
        roa_trend = institution_data.get('roa_trend', 0)
        if roa_trend < -0.5:
            warnings.append(f"Declining profitability trend")
        
        # Funding concentration
        concentration = institution_data.get('deposit_concentration', 0)
        if concentration > 20:
            warnings.append(f"High deposit concentration: {concentration:.1f}%")
        
        return warnings
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate prediction confidence"""
        # Confidence is higher for extreme probabilities
        return float(1 - 4 * (probability - 0.5) ** 2)
    
    def _generate_recommendations(self, risk_category: str, risk_drivers: List) -> List[str]:
        """Generate recommendations based on risk"""
        
        recommendations = []
        
        if risk_category in ["HIGH", "VERY_HIGH"]:
            recommendations.append("Immediate supervisory review required")
            recommendations.append("Consider capital injection requirements")
            recommendations.append("Enhance risk management controls")
        
        if any(driver['severity'] == 'HIGH' for driver in risk_drivers):
            recommendations.append("Address high-severity risk factors immediately")
        
        if risk_category == "MODERATE":
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Develop contingency plans")
        
        recommendations.append("Conduct stress testing for capital adequacy")
        recommendations.append("Review and update risk management framework")
        
        return recommendations
    
    def _initialize_models(self):
        """Initialize ML models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                random_state=42
            )
        }