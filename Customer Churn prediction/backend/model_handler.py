"""
Model Handler for FastAPI Backend
Handles model loading and prediction logic
"""

import joblib
import pandas as pd
import numpy as np
import os
import sys
import json

# Add src to path for preprocessor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import ChurnDataPreprocessor


class ChurnModelHandler:
    """Handles model loading and predictions"""
    
    def __init__(self, model_path='../models/churn_model.pkl', 
                 preprocessor_path='../models/preprocessor.pkl',
                 metrics_path='../models/model_metrics.json'):
        """Initialize and load model and preprocessor"""
        
        # Adjust paths relative to backend directory
        base_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(base_dir, model_path)
        self.preprocessor_path = os.path.join(base_dir, preprocessor_path)
        self.metrics_path = os.path.join(base_dir, metrics_path)
        
        self.model = None
        self.preprocessor = None
        self.metrics = None
        self.feature_importance = None
        
        self.load_model()
        self.load_metrics()
    
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = ChurnDataPreprocessor.load(self.preprocessor_path)
            print(f"✓ Model loaded successfully from {self.model_path}")
            print(f"✓ Preprocessor loaded successfully from {self.preprocessor_path}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def load_metrics(self):
        """Load model metrics"""
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                print(f"✓ Metrics loaded from {self.metrics_path}")
            
            # Load feature importance
            feature_imp_path = os.path.join(os.path.dirname(self.model_path), 'feature_importance.csv')
            if os.path.exists(feature_imp_path):
                self.feature_importance = pd.read_csv(feature_imp_path)
                print(f"✓ Feature importance loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load metrics: {str(e)}")
    
    def predict_single(self, customer_data: dict):
        """
        Make prediction for a single customer
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Dictionary with prediction and probability
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([customer_data])
            
            # Preprocess
            df_processed = self.preprocessor.transform_new_data(df)
            
            # Predict
            prediction = self.model.predict(df_processed)[0]
            probability = self.model.predict_proba(df_processed)[0]
            
            # Determine risk level
            churn_prob = probability[1]
            if churn_prob < 0.3:
                risk_level = "Low"
            elif churn_prob < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            result = {
                'churn': bool(prediction),
                'churn_probability': float(churn_prob),
                'retention_probability': float(probability[0]),
                'risk_level': risk_level,
                'prediction_label': 'Will Churn' if prediction else 'Will Stay'
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")
    
    def predict_batch(self, df: pd.DataFrame):
        """
        Make predictions for multiple customers
        
        Args:
            df: DataFrame with customer features
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Preprocess
            df_processed = self.preprocessor.transform_new_data(df.copy())
            
            # Predict
            predictions = self.model.predict(df_processed)
            probabilities = self.model.predict_proba(df_processed)
            
            # Add results to dataframe
            results_df = df.copy()
            results_df['Churn_Prediction'] = predictions
            results_df['Churn_Probability'] = probabilities[:, 1]
            results_df['Retention_Probability'] = probabilities[:, 0]
            results_df['Risk_Level'] = pd.cut(
                probabilities[:, 1],
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            results_df['Prediction_Label'] = results_df['Churn_Prediction'].map({
                0: 'Will Stay',
                1: 'Will Churn'
            })
            
            return results_df
            
        except Exception as e:
            raise ValueError(f"Batch prediction error: {str(e)}")
    
    def get_model_info(self):
        """Get model information and metrics"""
        info = {
            'model_loaded': self.model is not None,
            'preprocessor_loaded': self.preprocessor is not None,
        }
        
        if self.metrics:
            info['best_model'] = self.metrics.get('best_model', 'Unknown')
            info['metrics'] = self.metrics.get('all_results', {})
        
        if self.feature_importance is not None:
            info['top_features'] = self.feature_importance.head(10).to_dict('records')
        
        return info


# Global model handler instance
model_handler = None

def get_model_handler():
    """Get or create model handler singleton"""
    global model_handler
    if model_handler is None:
        model_handler = ChurnModelHandler()
    return model_handler
