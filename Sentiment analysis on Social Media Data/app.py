"""
Streamlit App for Sentiment Analysis Testing
This app provides an interactive interface to test the sentiment analysis model.
"""

import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        return model, vectorizer, preprocessor, model_info
    except FileNotFoundError:
        return None, None, None, None


def predict_sentiment(text, model, vectorizer, preprocessor):
    """
    Predict sentiment for input text
    
    Args:
        text (str): Input text
        model: Trained model
        vectorizer: TF-IDF vectorizer
        preprocessor: Text preprocessor
        
    Returns:
        tuple: (prediction, confidence)
    """
    # Preprocess
    cleaned = preprocessor.preprocess(text)
    
    # Vectorize
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    
    # Get probability scores
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100
    else:
        confidence = 0
    
    return prediction, confidence


def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment"""
    emoji_map = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emoji_map.get(sentiment, '❓')


def get_sentiment_color(sentiment):
    """Get color for sentiment"""
    color_map = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#ffc107'
    }
    return color_map.get(sentiment, '#6c757d')


def main():
    """Main Streamlit app"""
    
    # Header
    st.title("😊 Sentiment Analysis on Social Media Data")
    st.markdown("### Analyze sentiment from social media posts and reviews")
    st.markdown("---")
    
    # Load model
    model, vectorizer, preprocessor, model_info = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running `python train_model.py`")
        st.info("📝 Instructions:\n1. Open terminal\n2. Run: `python train_model.py`\n3. Wait for training to complete\n4. Refresh this page")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown(f"**Model:** {model_info['model_name']}")
        st.markdown(f"**Accuracy:** {model_info['accuracy']:.2%}")
        st.markdown(f"**Classes:** {', '.join(model_info['classes'])}")
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.markdown("""
        This app uses Natural Language Processing (NLP) to analyze sentiment in social media text.
        
        **Features:**
        - Single text prediction
        - Batch CSV upload
        - Confidence scores
        - Visual analytics
        """)
        
        # Display confusion matrix if exists
        if os.path.exists('confusion_matrix.png'):
            st.markdown("---")
            st.header("📈 Confusion Matrix")
            image = Image.open('confusion_matrix.png')
            st.image(image, use_column_width=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Analysis", "📊 Sample Data"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Analyze Single Text")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            user_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type or paste your social media post, review, or comment here...",
                help="Enter any text you want to analyze for sentiment"
            )
            
            analyze_button = st.button("🔍 Analyze Sentiment", key="analyze_single")
        
        with col2:
            st.markdown("### Quick Examples")
            examples = [
                "I love this product! It's amazing!",
                "This is terrible. Very disappointed.",
                "It's okay, nothing special."
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"Example {i+1}", key=f"example_{i}"):
                    user_input = example
                    st.session_state['example_text'] = example
        
        # Check if example was clicked
        if 'example_text' in st.session_state:
            user_input = st.session_state['example_text']
            del st.session_state['example_text']
        
        if analyze_button and user_input:
            with st.spinner("Analyzing..."):
                prediction, confidence = predict_sentiment(user_input, model, vectorizer, preprocessor)
                
                # Display result
                st.markdown("---")
                st.subheader("📊 Analysis Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    emoji = get_sentiment_emoji(prediction)
                    st.markdown(f"""
                    <div class="sentiment-{prediction}">
                        <h2 style="margin:0;">{emoji} {prediction.upper()}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if confidence > 0:
                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.progress(confidence / 100)
                
                # Visualization
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(vectorizer.transform([preprocessor.preprocess(user_input)]))[0]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Negative', 'Neutral', 'Positive'],
                            y=probabilities * 100,
                            marker_color=[get_sentiment_color('negative'), 
                                         get_sentiment_color('neutral'), 
                                         get_sentiment_color('positive')],
                            text=[f"{p:.1f}%" for p in probabilities * 100],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Sentiment Probability Distribution",
                        xaxis_title="Sentiment",
                        yaxis_title="Probability (%)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Analysis from CSV")
        st.markdown("Upload a CSV file with a column containing text to analyze")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Preview Data")
            st.dataframe(df.head())
            
            # Select text column
            text_column = st.selectbox("Select the column containing text:", df.columns)
            
            if st.button("🔍 Analyze All", key="analyze_batch"):
                with st.spinner(f"Analyzing {len(df)} texts..."):
                    predictions = []
                    confidences = []
                    
                    for text in df[text_column]:
                        pred, conf = predict_sentiment(str(text), model, vectorizer, preprocessor)
                        predictions.append(pred)
                        confidences.append(conf)
                    
                    df['sentiment'] = predictions
                    df['confidence'] = confidences
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display results
                    st.subheader("📊 Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution
                        sentiment_counts = df['sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color=sentiment_counts.index,
                            color_discrete_map={
                                'positive': get_sentiment_color('positive'),
                                'negative': get_sentiment_color('negative'),
                                'neutral': get_sentiment_color('neutral')
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Average confidence by sentiment
                        avg_confidence = df.groupby('sentiment')['confidence'].mean().reset_index()
                        fig = px.bar(
                            avg_confidence,
                            x='sentiment',
                            y='confidence',
                            title="Average Confidence by Sentiment",
                            color='sentiment',
                            color_discrete_map={
                                'positive': get_sentiment_color('positive'),
                                'negative': get_sentiment_color('negative'),
                                'neutral': get_sentiment_color('neutral')
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Sample Data
    with tab3:
        st.header("Sample Data for Testing")
        st.markdown("Use this sample data to test the batch analysis feature")
        
        from data_preprocessing import load_sample_data
        sample_df = load_sample_data()
        
        st.dataframe(sample_df)
        
        # Download sample data
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Data",
            data=csv,
            file_name="sample_social_media_data.csv",
            mime="text/csv"
        )
        
        # Sentiment distribution
        st.subheader("Sample Data Distribution")
        sentiment_counts = sample_df['sentiment'].value_counts()
        
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={'x': 'Sentiment', 'y': 'Count'},
            title="Sentiment Distribution in Sample Data",
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': get_sentiment_color('positive'),
                'negative': get_sentiment_color('negative'),
                'neutral': get_sentiment_color('neutral')
            }
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
