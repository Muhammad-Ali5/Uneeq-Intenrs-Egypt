# healthcare_diagnosis.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Since we don't have the actual dataset, I'll create synthetic data
def create_synthetic_data(n_samples=1000):
    """Create synthetic healthcare data for demonstration"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'blood_pressure': np.random.randint(90, 180, n_samples),
        'cholesterol': np.random.randint(150, 300, n_samples),
        'blood_sugar': np.random.randint(70, 200, n_samples),
        'bmi': np.random.uniform(18, 40, n_samples).round(1),
        'smoker': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'exercise': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),  # 0: None, 1: Moderate, 2: High
        'symptoms_duration': np.random.randint(1, 30, n_samples),
    }
    
    # Create diagnosis based on features (simplified rules)
    df = pd.DataFrame(data)
    
    # Simulate diagnosis probabilities
    risk_score = (
        (df['age'] > 60) * 1.5 +
        (df['blood_pressure'] > 140) * 1.2 +
        (df['cholesterol'] > 240) * 1.3 +
        (df['blood_sugar'] > 126) * 1.4 +
        df['smoker'] * 1.1 +
        df['family_history'] * 0.8 -
        df['exercise'] * 0.5
    )
    
    # Create diagnosis labels
    conditions = []
    for score in risk_score:
        if score < 2:
            conditions.append('Healthy')
        elif score < 4:
            conditions.append('Mild Condition')
        elif score < 6:
            conditions.append('Moderate Condition')
        else:
            conditions.append('Severe Condition')
    
    df['diagnosis'] = conditions
    
    return df

def train_model():
    """Train the diagnosis prediction model"""
    print("Creating synthetic healthcare data...")
    df = create_synthetic_data()
    
    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"\nDiagnosis Distribution:")
    print(df['diagnosis'].value_counts())
    
    # Encode target variable
    le = LabelEncoder()
    df['diagnosis_encoded'] = le.fit_transform(df['diagnosis'])
    
    # Prepare features and target
    features = ['age', 'blood_pressure', 'cholesterol', 'blood_sugar', 
                'bmi', 'smoker', 'family_history', 'exercise', 'symptoms_duration']
    X = df[features]
    y = df['diagnosis_encoded']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and encoders
    joblib.dump(model, 'healthcare_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    print("\nModel saved as 'healthcare_model.pkl'")
    print("Scaler saved as 'scaler.pkl'")
    print("Label encoder saved as 'label_encoder.pkl'")
    
    return model, scaler, le, features

if __name__ == "__main__":
    train_model()