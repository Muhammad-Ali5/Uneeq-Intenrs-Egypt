"""
Model Training Script for Customer Churn Prediction
Trains multiple classification models and saves the best one
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import json
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from data_preprocessing import ChurnDataPreprocessor


class ChurnModelTrainer:
    """Train and evaluate multiple models for churn prediction"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize multiple classification models"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=1,
                eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                class_weight='balanced',
                verbose=-1
            )
        }
        print(f"✓ Initialized {len(self.models)} models")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate their performance"""
        
        print("\n" + "="*60)
        print("MODEL TRAINING & EVALUATION")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n🔸 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'train_accuracy': (y_pred_train == y_train).mean(),
                'test_accuracy': (y_pred_test == y_test).mean(),
                'precision': precision_score(y_test, y_pred_test),
                'recall': recall_score(y_test, y_pred_test),
                'f1_score': f1_score(y_test, y_pred_test),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
            }
            
            self.results[name] = metrics
            
            # Print metrics
            print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
            print(f"  Precision:      {metrics['precision']:.4f}")
            print(f"  Recall:         {metrics['recall']:.4f}")
            print(f"  F1-Score:       {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
            print(f"  Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
        
        self._select_best_model()
        
    def _select_best_model(self):
        """Select the best model based on F1-score"""
        best_f1 = 0
        for name, metrics in self.results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"   F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}")
        print("="*60)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model"""
        if self.best_model_name == 'Logistic Regression':
            importance = np.abs(self.best_model.coef_[0])
        else:
            importance = self.best_model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, model_path='models/churn_model.pkl', 
                   metrics_path='models/model_metrics.json'):
        """Save the best model and metrics"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, model_path)
        print(f"\n✓ Saved best model to {model_path}")
        
        # Save metrics
        metrics_to_save = {
            'best_model': self.best_model_name,
            'all_results': self.results
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"✓ Saved metrics to {metrics_path}")
    
    @staticmethod
    def load_model(model_path='models/churn_model.pkl'):
        """Load a saved model"""
        return joblib.load(model_path)


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Step 1: Data Preprocessing
    print("\n[1/4] DATA PREPROCESSING")
    print("-" * 60)
    
    preprocessor = ChurnDataPreprocessor()
    
    # Check if dataset exists, if not create synthetic one
    data_path = 'data/raw/churn_data.csv'
    if not os.path.exists(data_path):
        print("Dataset not found. Creating synthetic dataset...")
        preprocessor.create_sample_dataset(n_samples=3000, save_path=data_path)
    
    # Prepare data with SMOTE
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        data_path,
        test_size=0.2,
        apply_smote=True
    )
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    
    # Step 2: Model Training
    print("\n[2/4] MODEL TRAINING")
    print("-" * 60)
    
    trainer = ChurnModelTrainer()
    trainer.initialize_models()
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Step 3: Feature Importance
    print("\n[3/4] FEATURE IMPORTANCE")
    print("-" * 60)
    
    feature_importance = trainer.get_feature_importance(preprocessor.feature_columns)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    print("\n✓ Saved feature importance to models/feature_importance.csv")
    
    # Step 4: Save Model
    print("\n[4/4] SAVING MODEL")
    print("-" * 60)
    
    trainer.save_model()
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("  - models/churn_model.pkl")
    print("  - models/preprocessor.pkl")
    print("  - models/model_metrics.json")
    print("  - models/feature_importance.csv")
    print("\nYou can now run the FastAPI backend and Streamlit frontend!")


if __name__ == "__main__":
    main()
