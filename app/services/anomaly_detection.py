import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import scipy.stats as stats
from typing import Dict, List, Tuple
import warnings
import joblib
warnings.filterwarnings('ignore')

class AdvancedAnomalyDetection:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_model = None # To hold the loaded model
        self.model_path = "ml_models/anomaly_detector.pkl"
        self._load_model()

    def _load_model(self):
        """Load pre-trained anomaly detection model"""
        try:
            if os.path.exists(self.model_path):
                self.anomaly_model = joblib.load(self.model_path)
                print(f"Loaded anomaly detection model from {self.model_path}")
            else:
                print(f"No pre-trained anomaly detection model found at {self.model_path}. Model will need to be trained or initialized.")
        except Exception as e:
            print(f"Error loading anomaly detection model: {e}")
            self.anomaly_model = None
    
    def detect_financial_anomalies(self, financial_data: pd.DataFrame, 
                                 institution_data: Dict = None) -> Dict:
        """Detect anomalies in financial data using multiple algorithms"""
        
        results = {}
        
        # 1. Statistical Outlier Detection
        statistical_anomalies = self._statistical_outlier_detection(financial_data)
        results['statistical'] = statistical_anomalies
        
        # 2. Machine Learning Anomaly Detection
        ml_anomalies = self._ml_anomaly_detection(financial_data)
        results['machine_learning'] = ml_anomalies
        
        # 3. Ratio-based Anomaly Detection
        ratio_anomalies = self._ratio_anomaly_detection(financial_data, institution_data)
        results['ratio_based'] = ratio_anomalies
        
        # 4. Temporal Anomaly Detection
        temporal_anomalies = self._temporal_anomaly_detection(financial_data)
        results['temporal'] = temporal_anomalies
        
        # 5. Consensus Scoring
        consensus_anomalies = self._consensus_scoring(results, financial_data)
        results['consensus'] = consensus_anomalies
        
        # 6. Risk Assessment
        risk_assessment = self._assess_anomaly_risk(consensus_anomalies, financial_data)
        results['risk_assessment'] = risk_assessment
        
        return results
    
    def _statistical_outlier_detection(self, data: pd.DataFrame) -> Dict:
        """Detect outliers using statistical methods"""
        
        anomalies = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            column_data = data[column].dropna()
            
            if len(column_data) < 3:
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(column_data))
            z_anomalies = column_data[z_scores > 3]
            
            # IQR method
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_anomalies = column_data[(column_data < (Q1 - 1.5 * IQR)) | 
                                      (column_data > (Q3 + 1.5 * IQR))]
            
            # Modified Z-score (robust to outliers)
            median = np.median(column_data)
            mad = np.median(np.abs(column_data - median))
            modified_z_scores = 0.6745 * (column_data - median) / mad if mad != 0 else 0
            mod_z_anomalies = column_data[np.abs(modified_z_scores) > 3.5]
            
            anomalies[column] = {
                'z_score_anomalies': len(z_anomalies),
                'iqr_anomalies': len(iqr_anomalies),
                'modified_z_anomalies': len(mod_z_anomalies),
                'anomaly_indices': list(set(
                    list(z_anomalies.index) +
                    list(iqr_anomalies.index) +
                    list(mod_z_anomalies.index)
                )),
                'anomaly_values': {
                    'z_score': z_anomalies.tolist(),
                    'iqr': iqr_anomalies.tolist(),
                    'modified_z': mod_z_anomalies.tolist()
                }
            }
        
        return anomalies
    
    def _ml_anomaly_detection(self, data: pd.DataFrame) -> Dict:
        """Detect anomalies using machine learning algorithms"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data) < 10:
            return {"error": "Insufficient data for ML anomaly detection"}
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.median())
        
        # Scale data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        ml_results = {}
        
        # Use loaded model if available, otherwise train
        if self.anomaly_model:
            iso_forest = self.anomaly_model
            print("Using pre-trained anomaly detection model.")
        else:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(scaled_data) # Train if not pre-trained
            print("No pre-trained model, training IsolationForest.")
            # Optionally save the trained model here if it was trained
            # joblib.dump(iso_forest, self.model_path)

        iso_predictions = iso_forest.fit_predict(scaled_data) # fit_predict is used here, which will re-fit.
                                                            # If the model is already fitted, just use predict.
                                                            # For IsolationForest, fit_predict is common.
        ml_results['isolation_forest'] = {
            'anomaly_indices': np.where(iso_predictions == -1)[0].tolist(),
            'anomaly_scores': iso_forest.decision_function(scaled_data).tolist()
        }
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=min(20, len(scaled_data)), contamination=0.1)
        lof_predictions = lof.fit_predict(scaled_data)
        ml_results['local_outlier_factor'] = {
            'anomaly_indices': np.where(lof_predictions == -1)[0].tolist(),
            'anomaly_scores': lof.negative_outlier_factor_.tolist()
        }
        
        # Elliptic Envelope (assuming Gaussian distribution)
        try:
            elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
            elliptic_predictions = elliptic.fit_predict(scaled_data)
            ml_results['elliptic_envelope'] = {
                'anomaly_indices': np.where(elliptic_predictions == -1)[0].tolist(),
                'anomaly_scores': elliptic.decision_function(scaled_data).tolist()
            }
        except:
            ml_results['elliptic_envelope'] = {"error": "Elliptic Envelope failed"}
        
        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)
        ml_results['dbscan'] = {
            'anomaly_indices': np.where(dbscan_labels == -1)[0].tolist(),
            'cluster_labels': dbscan_labels.tolist()
        }
        
        return ml_results
    
    def _ratio_anomaly_detection(self, data: pd.DataFrame, institution_data: Dict) -> Dict:
        """Detect anomalies in financial ratios"""
        
        ratio_anomalies = {}
        
        # Define normal ranges for key ratios (can be customized)
        ratio_thresholds = {
            'capital_adequacy_ratio': (10, 25),  # Minimum 10%, Strong up to 25%
            'npa_ratio': (0, 5),                # Maximum 5%
            'return_on_assets': (0.5, 3),       # Minimum 0.5%, Strong up to 3%
            'liquidity_ratio': (20, 40),        # Minimum 20%, Strong up to 40%
            'loan_to_deposit_ratio': (60, 80),  # Optimal range
            'cost_to_income_ratio': (40, 60)    # Maximum 60%
        }
        
        for ratio, (min_val, max_val) in ratio_thresholds.items():
            if ratio in data.columns:
                ratio_data = data[ratio].dropna()
                
                # Detect values outside normal range
                below_min = ratio_data[ratio_data < min_val]
                above_max = ratio_data[ratio_data > max_val]
                
                ratio_anomalies[ratio] = {
                    'below_threshold': {
                        'count': len(below_min),
                        'indices': below_min.index.tolist(),
                        'values': below_min.tolist()
                    },
                    'above_threshold': {
                        'count': len(above_max),
                        'indices': above_max.index.tolist(),
                        'values': above_max.tolist()
                    },
                    'threshold_range': [min_val, max_val]
                }
        
        return ratio_anomalies
    
    def _temporal_anomaly_detection(self, data: pd.DataFrame) -> Dict:
        """Detect temporal anomalies and trends"""
        
        temporal_anomalies = {}
        
        # Assuming data has a time index
        numeric_data = data.select_dtypes(include=[np.number])
        
        for column in numeric_data.columns:
            series = numeric_data[column].dropna()
            
            if len(series) < 4:
                continue
            
            # Detect sudden spikes/drops
            rolling_mean = series.rolling(window=3, center=True).mean()
            rolling_std = series.rolling(window=3, center=True).std()
            
            # Identify points 2 standard deviations from rolling mean
            z_scores = np.abs((series - rolling_mean) / rolling_std)
            temporal_outliers = series[z_scores > 2]
            
            # Detect trend breaks
            differences = series.diff().dropna()
            diff_z_scores = np.abs(stats.zscore(differences))
            trend_breaks = differences[diff_z_scores > 2.5]
            
            temporal_anomalies[column] = {
                'temporal_outliers': {
                    'count': len(temporal_outliers),
                    'indices': temporal_outliers.index.tolist(),
                    'values': temporal_outliers.tolist()
                },
                'trend_breaks': {
                    'count': len(trend_breaks),
                    'indices': trend_breaks.index.tolist(),
                    'values': trend_breaks.tolist()
                }
            }
        
        return temporal_anomalies
    
    def _consensus_scoring(self, individual_results: Dict, data: pd.DataFrame) -> Dict:
        """Combine results from multiple anomaly detection methods"""
        
        consensus_scores = {}
        record_count = len(data)
        
        for i in range(record_count):
            score = 0
            methods_contributing = 0
            
            # Statistical methods
            stat_anomalies = individual_results['statistical']
            for col, col_anomalies in stat_anomalies.items():
                if i in col_anomalies.get('anomaly_indices', []):
                    score += 1
                    methods_contributing += 1
            
            # ML methods
            ml_anomalies = individual_results['machine_learning']
            for method, method_results in ml_anomalies.items():
                if 'anomaly_indices' in method_results and i in method_results['anomaly_indices']:
                    score += 1
                    methods_contributing += 1
            
            # Ratio anomalies
            ratio_anomalies = individual_results['ratio_based']
            for ratio, ratio_result in ratio_anomalies.items():
                below_indices = ratio_result.get('below_threshold', {}).get('indices', [])
                above_indices = ratio_result.get('above_threshold', {}).get('indices', [])
                if i in below_indices or i in above_indices:
                    score += 1
                    methods_contributing += 1
            
            # Temporal anomalies
            temporal_anomalies = individual_results['temporal']
            for col, temp_results in temporal_anomalies.items():
                outlier_indices = temp_results.get('temporal_outliers', {}).get('indices', [])
                trend_indices = temp_results.get('trend_breaks', {}).get('indices', [])
                if i in outlier_indices or i in trend_indices:
                    score += 1
                    methods_contributing += 1
            
            # Normalize score
            if methods_contributing > 0:
                normalized_score = score / methods_contributing
            else:
                normalized_score = 0
            
            consensus_scores[i] = {
                'consensus_score': normalized_score,
                'methods_flagged': methods_contributing,
                'is_anomaly': normalized_score > 0.5  # Threshold for consensus
            }
        
        # Get top anomalies
        top_anomalies = sorted([(idx, score['consensus_score']) 
                              for idx, score in consensus_scores.items() 
                              if score['is_anomaly']], 
                             key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'consensus_scores': consensus_scores,
            'top_anomalies': top_anomalies,
            'total_anomalies': sum(1 for score in consensus_scores.values() if score['is_anomaly']),
            'anomaly_rate': sum(1 for score in consensus_scores.values() if score['is_anomaly']) / record_count
        }
    
    def _assess_anomaly_risk(self, consensus_results: Dict, data: pd.DataFrame) -> Dict:
        """Assess the risk level of detected anomalies"""
        
        total_records = len(data)
        anomaly_count = consensus_results['total_anomalies']
        anomaly_rate = consensus_results['anomaly_rate']
        
        # Risk assessment based on anomaly rate and severity
        if anomaly_rate == 0:
            risk_level = "LOW"
        elif anomaly_rate < 0.05:
            risk_level = "MODERATE"
        elif anomaly_rate < 0.15:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Get most common anomaly types
        top_anomalies = consensus_results['top_anomalies']
        
        risk_assessment = {
            'risk_level': risk_level,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'severity_assessment': self._get_severity_description(risk_level),
            'recommended_actions': self._get_anomaly_actions(risk_level),
            'top_anomalies_details': [
                {
                    'record_index': idx,
                    'anomaly_score': score,
                    'data_preview': data.iloc[idx].to_dict() if idx < len(data) else {}
                }
                for idx, score in top_anomalies
            ]
        }
        
        return risk_assessment
    
    def _get_severity_description(self, risk_level: str) -> str:
        """Get description for risk level"""
        descriptions = {
            "LOW": "Minimal anomalies detected. Normal operational variations.",
            "MODERATE": "Some anomalies present. Monitor for patterns.",
            "HIGH": "Significant anomalies detected. Investigation recommended.",
            "CRITICAL": "Critical anomaly levels. Immediate action required."
        }
        return descriptions.get(risk_level, "Unknown risk level")
    
    def _get_anomaly_actions(self, risk_level: str) -> List[str]:
        """Get recommended actions based on risk level"""
        actions = {
            "LOW": [
                "Continue regular monitoring",
                "Review anomaly detection thresholds periodically"
            ],
            "MODERATE": [
                "Increase monitoring frequency",
                "Investigate specific anomaly patterns",
                "Review data quality controls"
            ],
            "HIGH": [
                "Immediate investigation required",
                "Enhanced supervisory review",
                "Data validation and verification",
                "Root cause analysis"
            ],
            "CRITICAL": [
                "Immediate regulatory intervention",
                "Comprehensive audit required",
                "Risk mitigation plan activation",
                "Executive management review"
            ]
        }
        return actions.get(risk_level, [])
    