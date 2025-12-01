"""
Data Preprocessing Module for Customer Churn Prediction
Handles data cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os


class ChurnDataPreprocessor:
    """Handles all data preprocessing tasks for churn prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def create_sample_dataset(self, n_samples=1000, save_path='data/raw/churn_data.csv'):
        """
        Create a synthetic customer churn dataset for demonstration
        """
        np.random.seed(42)
        
        # Generate synthetic features
        data = {
            'CustomerID': [f'CUST{i:05d}' for i in range(n_samples)],
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'Tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples),
            'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
        }
        
        # Calculate TotalCharges
        data['TotalCharges'] = data['Tenure'] * data['MonthlyCharges'] + np.random.uniform(-100, 100, n_samples)
        data['TotalCharges'] = np.maximum(data['TotalCharges'], 0)
        
        # Create churn target (imbalanced: ~27% churn rate)
        # Higher churn probability for month-to-month contracts with low tenure
        churn_prob = np.where(
            (np.array(data['Contract']) == 'Month-to-month') & (np.array(data['Tenure']) < 12),
            0.45,  # High churn probability
            np.where(
                np.array(data['Contract']) == 'Two year',
                0.05,  # Low churn probability
                0.20   # Medium churn probability
            )
        )
        data['Churn'] = np.random.binomial(1, churn_prob).astype(str)
        data['Churn'] = np.where(data['Churn'] == '1', 'Yes', 'No')
        
        df = pd.DataFrame(data)
        
        # Save dataset
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Created synthetic dataset with {n_samples} samples")
        print(f"  Churn distribution: {df['Churn'].value_counts().to_dict()}")
        
        return df
    
    def load_and_clean_data(self, file_path):
        """Load and perform initial data cleaning"""
        df = pd.read_csv(file_path)
        
        # Handle TotalCharges (sometimes stored as string with spaces)
        if df['TotalCharges'].dtype == 'object':
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
        
        # Drop CustomerID as it's not useful for prediction
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        print(f"✓ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def engineer_features(self, df):
        """Create new features from existing ones"""
        df = df.copy()
        
        # Tenure grouping
        df['TenureGroup'] = pd.cut(df['Tenure'], 
                                   bins=[0, 12, 24, 48, 72], 
                                   labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
        
        # Average monthly charge per tenure month
        df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['Tenure'] + 1)
        
        # Service count
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        df['TotalServices'] = 0
        for col in service_cols:
            if col in df.columns:
                df['TotalServices'] += (df[col] == 'Yes').astype(int)
        
        print(f"✓ Engineered features: TenureGroup, AvgMonthlyCharge, TotalServices")
        return df
    
    def encode_features(self, df, is_training=True):
        """Encode categorical variables"""
        df = df.copy()
        
        # Separate target variable
        target = None
        if 'Churn' in df.columns:
            target = df['Churn'].map({'No': 0, 'Yes': 1})
            df = df.drop('Churn', axis=1)
        
        # Binary encoding for Yes/No columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'No': 0, 'Yes': 1})
        
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Label encoding for other categorical columns
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen labels
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        print(f"✓ Encoded {len(categorical_cols)} categorical features")
        return df, target
    
    def scale_features(self, df, is_training=True):
        """Scale numerical features"""
        df = df.copy()
        
        # Columns to scale
        scale_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge']
        scale_cols = [col for col in scale_cols if col in df.columns]
        
        if is_training:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        print(f"✓ Scaled {len(scale_cols)} numerical features")
        return df
    
    def prepare_data(self, file_path, test_size=0.2, apply_smote=True):
        """
        Complete preprocessing pipeline for training data
        """
        # Load and clean
        df = self.load_and_clean_data(file_path)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode
        df, target = self.encode_features(df, is_training=True)
        
        # Store feature columns
        self.feature_columns = df.columns.tolist()
        
        # Scale
        df = self.scale_features(df, is_training=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df, target, test_size=test_size, random_state=42, stratify=target
        )
        
        print(f"\n✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        print(f"  Original class distribution: {y_train.value_counts().to_dict()}")
        
        # Apply SMOTE to handle imbalance
        if apply_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  After SMOTE: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, df):
        """
        Transform new data using fitted preprocessor
        (for predictions)
        """
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode
        df, _ = self.encode_features(df, is_training=False)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the columns used in training
        df = df[self.feature_columns]
        
        # Scale
        df = self.scale_features(df, is_training=False)
        
        return df
    
    def save(self, filepath='models/preprocessor.pkl'):
        """Save the preprocessor"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"✓ Saved preprocessor to {filepath}")
    
    @staticmethod
    def load(filepath='models/preprocessor.pkl'):
        """Load a saved preprocessor"""
        return joblib.load(filepath)


if __name__ == "__main__":
    # Demo: Create sample dataset and preprocess
    preprocessor = ChurnDataPreprocessor()
    
    # Create synthetic dataset
    df = preprocessor.create_sample_dataset(n_samples=2000)
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        'data/raw/churn_data.csv',
        apply_smote=True
    )
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Feature shape: {X_train.shape}")
    print(f"  Features: {preprocessor.feature_columns[:5]}... (and {len(preprocessor.feature_columns)-5} more)")
    
    # Save preprocessor
    preprocessor.save()
