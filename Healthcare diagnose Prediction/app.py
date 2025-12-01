# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Healthcare Diagnosis Prediction",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Healthcare Diagnosis Prediction System")
st.markdown("""
This system predicts potential health conditions based on patient data using a machine learning model.
**Note:** This is a demonstration system using synthetic data. Always consult healthcare professionals for medical advice.
""")

# Sidebar for input
st.sidebar.header("Patient Information")

# Load model and encoders
@st.cache_resource
def load_model():
    try:
        model = joblib.load('healthcare_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, scaler, le
    except:
        st.error("Model files not found. Please run the training script first.")
        return None, None, None

model, scaler, le = load_model()

if model is not None:
    # Input features
    features = ['age', 'blood_pressure', 'cholesterol', 'blood_sugar', 
                'bmi', 'smoker', 'family_history', 'exercise', 'symptoms_duration']
    
    # Create input widgets
    inputs = {}
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        inputs['age'] = st.slider("Age", 20, 80, 45)
        inputs['blood_pressure'] = st.slider("Blood Pressure (mmHg)", 90, 180, 120)
        inputs['cholesterol'] = st.slider("Cholesterol (mg/dL)", 150, 300, 200)
        inputs['blood_sugar'] = st.slider("Blood Sugar (mg/dL)", 70, 200, 100)
    
    with col2:
        inputs['bmi'] = st.slider("BMI", 18.0, 40.0, 25.0, 0.1)
        inputs['smoker'] = st.selectbox("Smoker", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        inputs['family_history'] = st.selectbox("Family History of Heart Disease", 
                                                [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        inputs['exercise'] = st.selectbox("Exercise Level", 
                                          [("None", 0), ("Moderate", 1), ("High", 2)], 
                                          format_func=lambda x: x[0])[1]
    
    inputs['symptoms_duration'] = st.sidebar.slider("Symptoms Duration (days)", 1, 30, 7)
    
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Prediction button
    if st.sidebar.button("Predict Diagnosis", type="primary"):
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Decode prediction
        diagnosis = le.inverse_transform([prediction_encoded])[0]
        
        # Display results
        st.header("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Diagnosis", diagnosis)
        
        with col2:
            st.metric("Confidence", f"{max(prediction_proba)*100:.1f}%")
        
        with col3:
            risk_level = "Low" if diagnosis == "Healthy" else "Medium" if "Mild" in diagnosis else "High"
            st.metric("Risk Level", risk_level)
        
        # Probability distribution
        st.subheader("Probability Distribution")
        
        fig_prob = go.Figure(data=[
            go.Bar(
                x=le.classes_,
                y=prediction_proba * 100,
                marker_color=['#2E8B57', '#FFA500', '#FF8C00', '#FF4500']
            )
        ])
        
        fig_prob.update_layout(
            title="Diagnosis Probabilities",
            xaxis_title="Diagnosis",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Feature importance for this prediction
        st.subheader("Feature Impact Analysis")
        
        # Get feature importances
        feature_importance = model.feature_importances_
        
        fig_importance = go.Figure(data=[
            go.Bar(
                x=features,
                y=feature_importance * 100,
                marker_color='lightblue'
            )
        ])
        
        fig_importance.update_layout(
            title="Feature Importance in Model",
            xaxis_title="Features",
            yaxis_title="Importance (%)",
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Health metrics visualization
        st.subheader("Health Metrics Overview")
        
        # Create radar chart for health metrics
        categories = ['Age', 'BP', 'Cholesterol', 'Blood Sugar', 'BMI']
        
        # Normalize values for radar chart (0-1 scale)
        normalized_values = [
            (inputs['age'] - 20) / (80 - 20),
            1 - (inputs['blood_pressure'] - 90) / (180 - 90),  # Inverted (lower is better)
            1 - (inputs['cholesterol'] - 150) / (300 - 150),   # Inverted (lower is better)
            1 - (inputs['blood_sugar'] - 70) / (200 - 70),     # Inverted (lower is better)
            1 - (inputs['bmi'] - 18) / (40 - 18)               # Inverted (lower is better)
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],  # Close the shape
            theta=categories + [categories[0]],
            fill='toself',
            line_color='blue',
            name='Patient Metrics'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Health Metrics Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        
        recommendations = {
            "Healthy": [
                "✅ Maintain current lifestyle",
                "✅ Continue regular exercise",
                "✅ Annual health checkups recommended"
            ],
            "Mild Condition": [
                "⚠️ Monitor blood pressure regularly",
                "⚠️ Consider lifestyle modifications",
                "⚠️ Consult with a healthcare provider",
                "✅ Increase physical activity"
            ],
            "Moderate Condition": [
                "🔶 Schedule appointment with doctor",
                "🔶 Consider dietary changes",
                "🔶 Regular monitoring required",
                "🔶 Stress management recommended"
            ],
            "Severe Condition": [
                "🚨 Immediate medical consultation advised",
                "🚨 Follow-up with specialist required",
                "🚨 Comprehensive health assessment needed",
                "🚨 Consider treatment options"
            ]
        }
        
        for rec in recommendations.get(diagnosis, []):
            st.write(rec)
        
        # Download prediction results
        result_df = pd.DataFrame({
            'Feature': features + ['Predicted_Diagnosis', 'Confidence'],
            'Value': list(inputs.values()) + [diagnosis, f"{max(prediction_proba)*100:.1f}%"]
        })
        
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="health_prediction_results.csv",
            mime="text/csv"
        )
    
    else:
        st.info("👈 Enter patient information on the sidebar and click 'Predict Diagnosis'")
    
    # Model information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.info("""
    **Algorithm:** Random Forest Classifier
    **Features Used:** 9 patient parameters
    **Accuracy:** ~85% (on synthetic data)
    **Purpose:** Educational demonstration
    """)
    
    # Data preview
    with st.expander("View Sample Data"):
        from healthcare_diagnosis import create_synthetic_data
        sample_data = create_synthetic_data(10)
        st.dataframe(sample_data)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Important Disclaimer
    This tool is for educational and demonstration purposes only. It uses synthetic data and should NOT be used for actual medical diagnosis. 
    Always consult qualified healthcare professionals for medical advice and diagnosis.
    """)
    
else:
    st.error("""
    ### Model not found!
    Please run the training script first:
    ```
    python healthcare_diagnosis.py
    ```
    This will create the necessary model files.
    """)

# Instructions for running
with st.expander("Setup Instructions"):
    st.markdown("""
    ### How to set up and run this application:
    
    1. **Install required packages:**
    ```bash
    pip install streamlit pandas numpy scikit-learn joblib plotly
    ```
    
    2. **Train the model:**
    ```bash
    python healthcare_diagnosis.py
    ```
    
    3. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    
    4. **Access the app:** Open your browser and go to `http://localhost:8501`
    """)