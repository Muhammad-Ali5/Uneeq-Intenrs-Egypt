"""
Premium Streamlit Frontend for Customer Churn Prediction
Beautiful dark-themed UI with single and batch predictions
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import json

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css()
except:
    pass  # Fallback if CSS file not found

# API Configuration
API_URL = "http://localhost:8000"

# Helper Functions
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_prediction(customer_data):
    """Get single prediction from API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Title and Header
st.markdown("<h1>🎯 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #b4b4b4; font-size: 1.2rem; margin-bottom: 2rem;'>"
    "Predict customer churn using advanced machine learning • Powered by FastAPI & Streamlit"
    "</p>",
    unsafe_allow_html=True
)

# API Status Check
api_status = check_api_health()
if api_status:
    st.success("✅ API Connected")
else:
    st.error("❌ API Offline - Please start the FastAPI backend at http://localhost:8000")
    st.info("💡 Run: `cd backend && python main.py` or `uvicorn main:app --reload`")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Select Page",
        ["🔮 Single Prediction", "📁 Batch Prediction", "📈 Model Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This application predicts customer churn using machine learning.
    
    **Features:**
    - Single customer prediction
    - Batch CSV upload
    - Model performance metrics
    - Feature importance
    """)

# Page 1: Single Prediction
if page == "🔮 Single Prediction":
    st.markdown("## 🔮 Single Customer Prediction")
    st.markdown("Enter customer information to predict churn probability")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Customer Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            
            st.markdown("#### 📞 Phone Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        
        with col2:
            st.markdown("#### 🌐 Internet Services")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        with col3:
            st.markdown("#### 💳 Account Information")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=1.0)
        
        # Submit button
        submit_button = st.form_submit_button("🎯 Predict Churn", use_container_width=True)
    
    # Handle form submission
    if submit_button:
        # Prepare customer data
        customer_data = {
            "Gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        # Get prediction
        with st.spinner("🔮 Analyzing customer data..."):
            result = get_prediction(customer_data)
        
        if result:
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Prediction",
                    result['prediction_label'],
                    delta=None
                )
            
            with col2:
                churn_prob = result['churn_probability'] * 100
                st.metric(
                    "Churn Probability",
                    f"{churn_prob:.1f}%",
                    delta=f"{churn_prob - 50:.1f}% vs avg"
                )
            
            with col3:
                retention_prob = result['retention_probability'] * 100
                st.metric(
                    "Retention Probability",
                    f"{retention_prob:.1f}%",
                    delta=None
                )
            
            with col4:
                risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.metric(
                    "Risk Level",
                    f"{risk_color.get(result['risk_level'], '⚪')} {result['risk_level']}",
                    delta=None
                )
            
            # Visualize probabilities
            st.markdown("### 📈 Probability Distribution")
            fig = go.Figure(data=[
                go.Bar(
                    x=['Churn', 'Retention'],
                    y=[result['churn_probability'], result['retention_probability']],
                    marker_color=['#f5576c', '#4facfe'],
                    text=[f"{result['churn_probability']*100:.1f}%", f"{result['retention_probability']*100:.1f}%"],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(range=[0, 1], title="Probability"),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if result['churn']:
                st.warning(f"⚠️ **High Churn Risk Detected!**")
                st.markdown("""
                **Recommended Actions:**
                - 💰 Offer retention discount or loyalty reward
                - 📞 Schedule personalized customer service call
                - 🎁 Provide upgrade or additional services
                - 📧 Send targeted retention campaign
                """)
            else:
                st.success(f"✅ **Customer Likely to Stay**")
                st.markdown("""
                **Recommended Actions:**
                - 🌟 Continue excellent service
                - 📊 Monitor satisfaction regularly
                - 💝 Offer loyalty appreciation
                - 🔄 Consider upsell opportunities
                """)

# Page 2: Batch Prediction
elif page == "📁 Batch Prediction":
    st.markdown("## 📁 Batch Customer Prediction")
    st.markdown("Upload a CSV file with customer data to get predictions for multiple customers")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="CSV file must contain all required customer features"
    )
    
    if uploaded_file is not None:
        # Preview data
        df = pd.read_csv(uploaded_file)
        st.markdown(f"### 👀 Data Preview ({len(df)} rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Predict button
        if st.button("🎯 Predict All", use_container_width=True):
            with st.spinner(f"🔮 Processing {len(df)} predictions..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Send to API
                    files = {'file': uploaded_file}
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        files=files,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        # Parse results
                        results_df = pd.read_csv(StringIO(response.text))
                        
                        st.success(f"✅ Successfully predicted {len(results_df)} customers!")
                        
                        # Display summary metrics
                        st.markdown("### 📊 Prediction Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        churn_count = (results_df['Churn_Prediction'] == 1).sum()
                        churn_rate = (churn_count / len(results_df)) * 100
                        
                        with col1:
                            st.metric("Total Customers", len(results_df))
                        with col2:
                            st.metric("Will Churn", churn_count)
                        with col3:
                            st.metric("Will Stay", len(results_df) - churn_count)
                        with col4:
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")
                        
                        # Risk distribution
                        st.markdown("### 📈 Risk Distribution")
                        risk_counts = results_df['Risk_Level'].value_counts()
                        
                        fig = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            color=risk_counts.index,
                            color_discrete_map={'Low': '#4facfe', 'Medium': '#ffc107', 'High': '#f5576c'}
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error during batch prediction: {str(e)}")
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Show sample format
        with st.expander("📝 View Required CSV Format"):
            st.markdown("""
            Your CSV file must include the following columns:
            - Gender, SeniorCitizen, Partner, Dependents
            - Tenure, PhoneService, MultipleLines
            - InternetService, OnlineSecurity, OnlineBackup
            - DeviceProtection, TechSupport, StreamingTV, StreamingMovies
            - Contract, PaperlessBilling, PaymentMethod
            - MonthlyCharges, TotalCharges
            """)

# Page 3: Model Insights
elif page == "📈 Model Insights":
    st.markdown("## 📈 Model Performance & Insights")
    
    # Get model info
    with st.spinner("📊 Loading model information..."):
        model_info = get_model_info()
    
    if model_info:
        # Model details
        st.markdown("### 🤖 Model Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Best Model:** {model_info.get('best_model', 'N/A')}")
        with col2:
            status = "✅ Loaded" if model_info.get('model_loaded') else "❌ Not Loaded"
            st.info(f"**Status:** {status}")
        
        # Performance metrics
        if 'metrics' in model_info:
            st.markdown("### 📊 Model Performance Comparison")
            
            metrics_data = []
            for model_name, metrics in model_info['metrics'].items():
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['test_accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'ROC-AUC': f"{metrics['roc_auc']:.4f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model metrics visualization
            best_model = model_info.get('best_model')
            if best_model in model_info['metrics']:
                st.markdown(f"### 🏆 {best_model} - Detailed Metrics")
                
                best_metrics = model_info['metrics'][best_model]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{best_metrics['test_accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{best_metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{best_metrics['recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{best_metrics['f1_score']:.4f}")
                
                # Confusion Matrix
                if 'confusion_matrix' in best_metrics:
                    st.markdown("#### 📊 Confusion Matrix")
                    cm = best_metrics['confusion_matrix']
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted: Stay', 'Predicted: Churn'],
                        y=['Actual: Stay', 'Actual: Churn'],
                        colorscale='Viridis',
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 16}
                    ))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if 'top_features' in model_info:
            st.markdown("### 🎯 Top 10 Important Features")
            
            features_df = pd.DataFrame(model_info['top_features'])
            
            fig = px.bar(
                features_df,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis={'categoryorder': 'total ascending'},
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not load model information. Ensure the model is trained and API is running.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #b4b4b4;'>"
    "Built with ❤️ using FastAPI & Streamlit | Customer Churn Prediction System"
    "</p>",
    unsafe_allow_html=True
)
