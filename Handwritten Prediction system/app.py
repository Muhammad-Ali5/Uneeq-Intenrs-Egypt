"""
Handwritten Digit Recognition - Streamlit Web Application
Clean and simple interface with accurate predictions for single and multiple digits
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_drawable_canvas import st_canvas
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean CSS
st.markdown("""
<style>
    /* Clean color scheme */
    .main-header {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .info-box {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
    
    .prediction-result {
        background: #2563eb;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-digit {
        font-size: 4rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .stButton>button {
        background: #2563eb;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: #1d4ed8;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model and metrics
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model('digit_recognition_model.h5')
        return model
    except:
        try:
            model = keras.models.load_model('digit_recognition_model')
            return model
        except:
            return None

@st.cache_data
def load_metrics():
    """Load model metrics"""
    try:
        with open('model_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

def process_single_digit(img_gray, x, y, w, h):
    """Process a single digit crop"""
    # Add padding
    padding = 20
    # Ensure we don't go out of bounds
    y_start = max(0, y - padding)
    y_end = min(img_gray.shape[0], y + h + padding)
    x_start = max(0, x - padding)
    x_end = min(img_gray.shape[1], x + w + padding)
    
    # Crop
    img_cropped = img_gray[y_start:y_end, x_start:x_end]
    
    # Make square
    h_c, w_c = img_cropped.shape
    size = max(w_c, h_c)
    img_square = np.zeros((size, size), dtype=np.uint8)
    
    # Center
    y_offset = (size - h_c) // 2
    x_offset = (size - w_c) // 2
    img_square[y_offset:y_offset+h_c, x_offset:x_offset+w_c] = img_cropped
    
    # Resize to 28x28
    img_resized = cv2.resize(img_square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Reshape
    img_input = img_normalized.reshape(1, 784)
    
    return img_input, img_resized

def get_contours(img_gray):
    """Find and sort contours from left to right"""
    # Threshold to ensure binary image
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small noise
    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:  # Minimum area threshold
            valid_contours.append(cnt)
            
    # Sort contours from left to right
    if valid_contours:
        bounding_boxes = [cv2.boundingRect(c) for c in valid_contours]
        (valid_contours, bounding_boxes) = zip(*sorted(zip(valid_contours, bounding_boxes),
                                                      key=lambda b: b[1][0]))
        return valid_contours, bounding_boxes
    return [], []

# Initialize model
model = load_model()
metrics = load_metrics()

# Sidebar
with st.sidebar:
    st.markdown("### 🔢 Navigation")
    page = st.radio(
        "",
        ["🏠 Home", "✏️ Draw & Predict", "📊 Batch Prediction", "📈 Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if model:
        st.success("✅ Model Ready")
        if metrics:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    else:
        st.error("❌ Model Not Found")
        st.info("Run: `python train_model.py`")

# Page: Home
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1>🔢 Handwritten Digit Recognition</h1>
        <p>Deep Learning with 98%+ Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>98%+</div>
            <div class='metric-label'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>Multi</div>
            <div class='metric-label'>Digit Support</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>MNIST</div>
            <div class='metric-label'>Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>📝 About This Project</h3>
        <p>A deep learning system that recognizes handwritten digits using a Multi-layer Perceptron 
        trained on the MNIST dataset. The model achieves over 98% accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>✨ Features</h3>
        <ul>
            <li><strong>Draw & Predict:</strong> Draw single or multiple digits (e.g., "100")</li>
            <li><strong>Batch Processing:</strong> Upload CSV files for multiple predictions</li>
            <li><strong>High Accuracy:</strong> 98%+ accuracy on test data</li>
            <li><strong>Performance Metrics:</strong> View detailed model statistics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Page: Draw & Predict
elif page == "✏️ Draw & Predict":
    st.markdown("""
    <div class='main-header'>
        <h1>✏️ Draw & Predict</h1>
        <p>Draw one or more digits (e.g., 1, 5, 123)</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not model:
        st.error("❌ Model not loaded. Please train the model first by running `train_model.py`")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎨 Draw Here")
            st.info("✏️ Draw clearly. You can draw multiple digits!")
            
            # Canvas
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=15,
                stroke_color="white",
                background_color="black",
                height=280,
                width=500,  # Wider canvas for multiple digits
                drawing_mode="freedraw",
                key="canvas",
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_button = st.button("🔮 Predict", use_container_width=True)
            with col_btn2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.rerun()
        
        with col2:
            st.markdown("### 🎯 Prediction")
            
            if predict_button and canvas_result.image_data is not None:
                # Check if canvas has content
                if np.sum(canvas_result.image_data[:,:,3]) > 0:
                    # Get image and convert alpha to grayscale
                    img = canvas_result.image_data.astype('uint8')
                    alpha = img[:, :, 3]
                    img_gray = alpha  # Use alpha channel directly
                    
                    # Find contours (digits)
                    contours, boxes = get_contours(img_gray)
                    
                    if not contours:
                        st.warning("Could not detect any digits. Try drawing thicker lines.")
                    else:
                        predictions = []
                        confidences = []
                        processed_images = []
                        
                        # Process each digit
                        for box in boxes:
                            x, y, w, h = box
                            img_input, img_resized = process_single_digit(img_gray, x, y, w, h)
                            
                            # Predict
                            pred = model.predict(img_input, verbose=0)
                            digit = np.argmax(pred[0])
                            conf = pred[0][digit] * 100
                            
                            predictions.append(str(digit))
                            confidences.append(conf)
                            processed_images.append(img_resized)
                        
                        # Display combined result
                        final_result = "".join(predictions)
                        avg_conf = sum(confidences) / len(confidences)
                        
                        st.markdown(f"""
                        <div class='prediction-result'>
                            <h3>Predicted Number</h3>
                            <div class='prediction-digit'>{final_result}</div>
                            <p style='font-size: 1.2rem;'>Avg Confidence: {avg_conf:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show individual digits
                        st.markdown("### 🔍 Individual Digits")
                        cols = st.columns(len(predictions))
                        for i, (digit, conf, img) in enumerate(zip(predictions, confidences, processed_images)):
                            with cols[i]:
                                st.image(img, width=100, caption=f"Digit: {digit} ({conf:.0f}%)")

                else:
                    st.warning("⚠️ Please draw something first!")
            else:
                st.markdown("""
                <div class='info-box' style='text-align: center; padding: 3rem;'>
                    <h3>👈 Draw digits</h3>
                    <p>Draw any number (e.g., 7, 42, 100)</p>
                    <p>The AI will detect and recognize each digit separately.</p>
                </div>
                """, unsafe_allow_html=True)

# Page: Batch Prediction
elif page == "📊 Batch Prediction":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Batch Prediction</h1>
        <p>Upload CSV for multiple predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not model:
        st.error("❌ Model not loaded. Please train the model first.")
    else:
        st.markdown("""
        <div class='info-box'>
            <h3>📋 CSV Format</h3>
            <ul>
                <li>Each row = one digit image (784 pixel values)</li>
                <li>Pixel values: 0-255</li>
                <li>Optional 'label' column for actual digits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {df.shape[0]} samples")
                
                has_labels = 'label' in df.columns
                
                if has_labels:
                    X_test = df.drop('label', axis=1).values
                    y_test = df['label'].values
                else:
                    X_test = df.values
                    y_test = None
                
                X_test = X_test / 255.0
                
                max_samples = st.slider("Samples to predict", 10, min(1000, len(X_test)), 100)
                X_sample = X_test[:max_samples]
                
                if st.button("🚀 Run Prediction", use_container_width=True):
                    with st.spinner(f"Predicting {max_samples} samples..."):
                        predictions = model.predict(X_sample, verbose=0)
                        predicted_classes = np.argmax(predictions, axis=1)
                    
                    st.success(f"✅ Completed!")
                    
                    results_df = pd.DataFrame({
                        'Sample': range(1, max_samples + 1),
                        'Predicted': predicted_classes,
                        'Confidence (%)': [predictions[i][predicted_classes[i]] * 100 for i in range(max_samples)]
                    })
                    
                    if has_labels:
                        y_sample = y_test[:max_samples]
                        results_df['Actual'] = y_sample
                        results_df['Correct'] = results_df['Predicted'] == results_df['Actual']
                        
                        accuracy = (results_df['Correct'].sum() / max_samples) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", max_samples)
                        with col2:
                            st.metric("Correct", results_df['Correct'].sum())
                        with col3:
                            st.metric("Accuracy", f"{accuracy:.2f}%")
                    
                    st.dataframe(results_df, use_container_width=True, height=400)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    # Sample images
                    st.markdown("### 🖼️ Sample Predictions")
                    cols = st.columns(5)
                    for i in range(min(10, max_samples)):
                        with cols[i % 5]:
                            img = X_sample[i].reshape(28, 28)
                            fig, ax = plt.subplots(figsize=(2, 2))
                            ax.imshow(img, cmap='gray')
                            ax.axis('off')
                            title = f"Pred: {predicted_classes[i]}"
                            if has_labels:
                                title += f"\nTrue: {y_sample[i]}"
                            ax.set_title(title, fontsize=8)
                            st.pyplot(fig)
                            plt.close()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Page: Performance
elif page == "📈 Performance":
    st.markdown("""
    <div class='main-header'>
        <h1>📈 Model Performance</h1>
        <p>Detailed metrics and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not metrics:
        st.warning("⚠️ Metrics not found. Train the model first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{metrics['accuracy']*100:.2f}%</div>
                <div class='metric-label'>Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{metrics['loss']:.4f}</div>
                <div class='metric-label'>Loss</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{metrics['training_samples']:,}</div>
                <div class='metric-label'>Training</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{metrics['epochs_trained']}</div>
                <div class='metric-label'>Epochs</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confusion Matrix
        try:
            st.markdown("### 🎯 Confusion Matrix")
            cm = np.array(metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
            ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            st.pyplot(fig)
            plt.close()
        except:
            pass
        
        # Training plots
        try:
            st.markdown("### 📊 Training History")
            col1, col2 = st.columns(2)
            with col1:
                st.image('training_history.png', use_container_width=True)
            with col2:
                st.image('confusion_matrix.png', use_container_width=True)
        except:
            pass
        
        # Classification report
        if 'classification_report' in metrics:
            st.markdown("### 📋 Per-Digit Performance")
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            report_df = report_df.iloc[:10]  # Only digits 0-9
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# Page: About
else:
    st.markdown("""
    <div class='main-header'>
        <h1>ℹ️ About</h1>
        <p>Project Information</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>🎯 Project</h3>
        <p>Handwritten Digit Recognition using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>🧠 Model</h3>
        <p><strong>Multi-layer Perceptron</strong></p>
        <ul>
            <li>Input: 784 neurons (28×28 pixels)</li>
            <li>Hidden: 512 → 256 → 128 → 64 neurons</li>
            <li>Output: 10 neurons (digits 0-9)</li>
            <li>Activation: ReLU + Softmax</li>
            <li>Regularization: Dropout + BatchNorm</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>🛠️ Tech Stack</h3>
        <ul>
            <li><strong>Deep Learning:</strong> TensorFlow & Keras</li>
            <li><strong>Web App:</strong> Streamlit</li>
            <li><strong>Data:</strong> NumPy, Pandas</li>
            <li><strong>Visualization:</strong> Matplotlib, Seaborn</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>📊 Dataset</h3>
        <p><strong>MNIST</strong> - 70,000 handwritten digit images</p>
        <ul>
            <li>60,000 training images</li>
            <li>10,000 test images</li>
            <li>28×28 grayscale</li>
            <li>10 classes (0-9)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>👨‍💻 Developer</h3>
        <p>Built for <strong>#UneeQinterns</strong></p>
    </div>
    """, unsafe_allow_html=True)
