"""
Machine Learning Model Training Module
Trains Random Forest, Logistic Regression, and Isolation Forest models for bank risk prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
from database import SessionLocal
from models import Bank, CAMELSRating, RiskScore
from datetime import datetime

MODEL_DIR = 'ml_models'
os.makedirs(MODEL_DIR, exist_ok=True)

class MLModelTrainer:
    
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
    
    def extract_training_data(self):
        """Extract training data from database"""
        db = SessionLocal()
        
        try:
            camels_ratings = db.query(CAMELSRating).all()
            risk_scores = db.query(RiskScore).all()
            
            if len(camels_ratings) == 0:
                print("No CAMELS data found. Creating synthetic training data...")
                return self._create_synthetic_data()
            
            training_data = []
            
            for rating in camels_ratings:
                # Extract features
                features = {
                    'capital_adequacy_ratio': float(rating.capital_adequacy_ratio or 12.0),
                    'npl_ratio': float(rating.npl_ratio or 3.0),
                    'roa': float(rating.return_on_assets or 1.0),
                    'roe': float(rating.return_on_equity or 10.0),
                    'liquidity_ratio': float(rating.liquidity_ratio or 25.0),
                    'loan_to_deposit_ratio': float(rating.loan_to_deposit_ratio or 85.0),
                    'cost_to_income': 60.0,  # Default value
                    'deposit_growth': 5.0,   # Default value
                    'loan_growth': 5.0,       # Default value
                    'equity_to_assets': 10.0  # Default value
                }
                
                # Determine if bank "failed" based on composite rating (4-5 = high risk/failed)
                failed = 1 if rating.composite_rating >= 4 else 0
                
                # Add PD based on rating
                pd_value = self._rating_to_pd(rating.composite_rating)
                
                features['failed'] = failed
                features['pd_target'] = pd_value
                
                training_data.append(features)
            
            # If we have real data, augment it with synthetic data to ensure enough samples
            df = pd.DataFrame(training_data)
            
            if len(df) < 50:
                print(f"Only {len(df)} real samples. Adding synthetic data...")
                synthetic_df = self._create_synthetic_data()
                df = pd.concat([df, synthetic_df], ignore_index=True)
            
            print(f"Training data prepared: {len(df)} samples")
            print(f"Failed banks: {df['failed'].sum()}, Healthy banks: {len(df) - df['failed'].sum()}")
            
            return df
            
        finally:
            db.close()
    
    def _rating_to_pd(self, composite_rating):
        """Convert CAMELS composite rating to PD"""
        rating_pd_map = {
            1: 0.0010,
            2: 0.0050,
            3: 0.0200,
            4: 0.0800,
            5: 0.2000
        }
        return rating_pd_map.get(composite_rating, 0.05)
    
    def _create_synthetic_data(self, n_samples=100):
        """Create synthetic training data for ML models"""
        np.random.seed(42)
        
        data = []
        
        # Generate healthy banks (70%)
        n_healthy = int(n_samples * 0.7)
        for i in range(n_healthy):
            sample = {
                'capital_adequacy_ratio': np.random.normal(14, 2),
                'npl_ratio': np.random.gamma(2, 1.5),
                'roa': np.random.normal(1.5, 0.4),
                'roe': np.random.normal(14, 3),
                'liquidity_ratio': np.random.normal(28, 4),
                'loan_to_deposit_ratio': np.random.normal(82, 6),
                'cost_to_income': np.random.normal(58, 8),
                'deposit_growth': np.random.normal(6, 3),
                'loan_growth': np.random.normal(6, 3),
                'equity_to_assets': np.random.normal(11, 2),
                'failed': 0,
                'pd_target': np.random.uniform(0.001, 0.01)
            }
            data.append(sample)
        
        # Generate stressed/failed banks (30%)
        n_stressed = n_samples - n_healthy
        for i in range(n_stressed):
            sample = {
                'capital_adequacy_ratio': np.random.normal(9, 2),
                'npl_ratio': np.random.gamma(4, 2),
                'roa': np.random.normal(0.3, 0.5),
                'roe': np.random.normal(3, 4),
                'liquidity_ratio': np.random.normal(18, 4),
                'loan_to_deposit_ratio': np.random.normal(95, 8),
                'cost_to_income': np.random.normal(78, 10),
                'deposit_growth': np.random.normal(-2, 5),
                'loan_growth': np.random.normal(-1, 4),
                'equity_to_assets': np.random.normal(7, 2),
                'failed': 1,
                'pd_target': np.random.uniform(0.05, 0.25)
            }
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        # Clip values to realistic ranges
        df['capital_adequacy_ratio'] = df['capital_adequacy_ratio'].clip(5, 25)
        df['npl_ratio'] = df['npl_ratio'].clip(0.5, 25)
        df['roa'] = df['roa'].clip(-2, 5)
        df['roe'] = df['roe'].clip(-10, 30)
        df['liquidity_ratio'] = df['liquidity_ratio'].clip(10, 50)
        df['loan_to_deposit_ratio'] = df['loan_to_deposit_ratio'].clip(50, 120)
        df['cost_to_income'] = df['cost_to_income'].clip(40, 100)
        df['deposit_growth'] = df['deposit_growth'].clip(-20, 30)
        df['loan_growth'] = df['loan_growth'].clip(-20, 30)
        df['equity_to_assets'] = df['equity_to_assets'].clip(5, 20)
        
        return df
    
    def train_all_models(self):
        """Train all ML models"""
        print("\n=== Starting ML Model Training ===")
        
        # Extract training data
        training_data = self.extract_training_data()
        
        if training_data is None or len(training_data) == 0:
            print("ERROR: No training data available")
            return False
        
        # Prepare features and labels
        X = training_data[self.feature_names].fillna(0)
        y_failure = training_data['failed'].fillna(0)
        y_pd = training_data['pd_target'].fillna(0.05)
        
        # Train-test split
        X_train, X_test, y_fail_train, y_fail_test = train_test_split(
            X, y_failure, test_size=0.2, random_state=42, 
            stratify=y_failure if len(y_failure.unique()) > 1 else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest for failure prediction
        print("\nTraining Random Forest failure prediction model...")
        self.failure_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        self.failure_model.fit(X_train_scaled, y_fail_train)
        
        train_acc = self.failure_model.score(X_train_scaled, y_fail_train)
        test_acc = self.failure_model.score(X_test_scaled, y_fail_test)
        cv_scores = cross_val_score(self.failure_model, X_train_scaled, y_fail_train, cv=5)
        
        print(f"Random Forest - Train Accuracy: {train_acc:.4f}")
        print(f"Random Forest - Test Accuracy: {test_acc:.4f}")
        print(f"Random Forest - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        feature_importance = dict(zip(self.feature_names, self.failure_model.feature_importances_))
        print("Feature Importance:", {k: f"{v:.4f}" for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]})
        
        # Train Logistic Regression for PD prediction
        print("\nTraining Logistic Regression PD model...")
        X_train_pd, X_test_pd, y_pd_train, y_pd_test = train_test_split(
            X, y_pd, test_size=0.2, random_state=42
        )
        X_train_pd_scaled = self.scaler.fit_transform(X_train_pd)
        X_test_pd_scaled = self.scaler.transform(X_test_pd)
        
        # Convert continuous PD to binary for classification
        y_pd_binary_train = (y_pd_train > 0.02).astype(int)
        y_pd_binary_test = (y_pd_test > 0.02).astype(int)
        
        self.pd_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        self.pd_model.fit(X_train_pd_scaled, y_pd_binary_train)
        
        pd_train_acc = self.pd_model.score(X_train_pd_scaled, y_pd_binary_train)
        pd_test_acc = self.pd_model.score(X_test_pd_scaled, y_pd_binary_test)
        
        print(f"Logistic Regression PD - Train Accuracy: {pd_train_acc:.4f}")
        print(f"Logistic Regression PD - Test Accuracy: {pd_test_acc:.4f}")
        
        # Train Isolation Forest for anomaly detection
        print("\nTraining Isolation Forest anomaly detector...")
        self.anomaly_detector = IsolationForest(
            contamination=0.15,
            random_state=42,
            n_estimators=100
        )
        self.anomaly_detector.fit(X)
        
        anomalies = self.anomaly_detector.predict(X)
        n_anomalies = (anomalies == -1).sum()
        print(f"Isolation Forest - Detected {n_anomalies} anomalies out of {len(X)} samples ({n_anomalies/len(X)*100:.2f}%)")
        
        # Save models
        print("\nSaving trained models...")
        self.save_models()
        
        print("\n=== ML Model Training Complete ===\n")
        return True
    
    def save_models(self):
        """Save trained models to disk"""
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        joblib.dump(self.failure_model, os.path.join(MODEL_DIR, 'failure_model.pkl'))
        joblib.dump(self.pd_model, os.path.join(MODEL_DIR, 'pd_model.pkl'))
        joblib.dump(self.anomaly_detector, os.path.join(MODEL_DIR, 'anomaly_detector.pkl'))
        
        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'model_versions': {
                'failure_model': 'RandomForestClassifier_v1',
                'pd_model': 'LogisticRegression_v1',
                'anomaly_detector': 'IsolationForest_v1'
            }
        }
        joblib.dump(metadata, os.path.join(MODEL_DIR, 'metadata.pkl'))
        
        print(f"Models saved to {MODEL_DIR}/")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            self.failure_model = joblib.load(os.path.join(MODEL_DIR, 'failure_model.pkl'))
            self.pd_model = joblib.load(os.path.join(MODEL_DIR, 'pd_model.pkl'))
            self.anomaly_detector = joblib.load(os.path.join(MODEL_DIR, 'anomaly_detector.pkl'))
            
            print("ML models loaded successfully from disk")
            return True
        except Exception as e:
            print(f"Could not load models from disk: {e}")
            return False

def train_models():
    """Main function to train all ML models"""
    trainer = MLModelTrainer()
    success = trainer.train_all_models()
    return success

if __name__ == '__main__':
    train_models()
