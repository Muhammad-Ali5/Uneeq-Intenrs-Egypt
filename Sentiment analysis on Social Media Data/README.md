# 😊 Sentiment Analysis on Social Media Data

A simple and easy-to-understand sentiment analysis system that classifies social media posts and reviews as **positive**, **negative**, or **neutral** using Natural Language Processing (NLP) techniques.

## 📋 Project Overview

This project performs sentiment analysis on social media data using:
- **Text Preprocessing**: Cleaning, tokenization, lemmatization
- **Feature Extraction**: TF-IDF vectorization
- **Machine Learning Models**: Logistic Regression, Naive Bayes, Random Forest
- **Interactive UI**: Streamlit web application for testing

## 🚀 Features

- ✅ Clean and simple code structure
- ✅ Multiple ML models with automatic best model selection
- ✅ Interactive Streamlit app for testing
- ✅ Single text prediction
- ✅ Batch CSV file analysis
- ✅ Visual analytics and confidence scores
- ✅ Sample data included for testing

## 📁 Project Structure

```
Sentiment analysis on Social Media Data/
├── data_preprocessing.py    # Text preprocessing utilities
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── sentiment_model.pkl    # Trained model (generated)
├── vectorizer.pkl         # TF-IDF vectorizer (generated)
├── preprocessor.pkl       # Text preprocessor (generated)
├── model_info.pkl         # Model metadata (generated)
└── confusion_matrix.png   # Model evaluation plot (generated)
```

## 🛠️ Installation

### Step 1: Clone or Navigate to Project Directory

```bash
cd "e:\Interships\Uneeq intern Egypt\Sentiment analysis on Social Media Data"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data (Automatic)

The required NLTK data will be downloaded automatically when you run the scripts for the first time.

## 📚 Usage

### 1️⃣ Train the Model

First, train the sentiment analysis model:

```bash
python train_model.py
```

This will:
- Load sample social media data
- Preprocess the text
- Train multiple models (Logistic Regression, Naive Bayes, Random Forest)
- Evaluate and select the best model
- Save the trained model and preprocessing objects
- Generate a confusion matrix visualization

**Expected Output:**
```
Training samples: 20
Testing samples: 5
Training Logistic Regression...
Accuracy: 0.XXXX
...
Best Model: [Model Name]
Best Accuracy: 0.XXXX
```

### 2️⃣ Run the Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3️⃣ Use the App

The Streamlit app has three main features:

#### 🔍 Single Prediction
- Enter any text in the text area
- Click "Analyze Sentiment"
- View the predicted sentiment with confidence score
- See probability distribution chart

#### 📁 Batch Analysis
- Upload a CSV file with text data
- Select the column containing text
- Click "Analyze All"
- Download results as CSV
- View sentiment distribution charts

#### 📊 Sample Data
- View sample social media data
- Download sample CSV for testing
- See data distribution

## 📊 Model Details

### Preprocessing Steps
1. Convert text to lowercase
2. Remove URLs, mentions (@user), and hashtags
3. Remove special characters and numbers
4. Tokenization
5. Remove stopwords
6. Lemmatization
7. TF-IDF vectorization

### Models Trained
- **Logistic Regression**: Linear model for classification
- **Naive Bayes**: Probabilistic classifier
- **Random Forest**: Ensemble learning method

The system automatically selects the best performing model based on accuracy.

### Evaluation Metrics
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix

## 📝 Example Usage

### Python Code Example

```python
from data_preprocessing import TextPreprocessor
import pickle

# Load model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Predict sentiment
text = "I love this product! It's amazing!"
cleaned = preprocessor.preprocess(text)
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)[0]

print(f"Sentiment: {prediction}")  # Output: positive
```

### CSV Format for Batch Analysis

Your CSV file should have at least one column with text data:

```csv
text
"I love this product! It's amazing!"
"Terrible experience. Very disappointed."
"It's okay, nothing special."
```

## 🎯 Sample Data

The project includes sample social media data for testing. You can:
- View it in the "Sample Data" tab of the Streamlit app
- Download it as CSV
- Use it to test batch analysis

## 🔧 Customization

### Using Your Own Dataset

Replace the sample data in `train_model.py`:

```python
# Instead of using load_sample_data()
df = pd.read_csv('your_dataset.csv')
```

Your dataset should have:
- A column with text data
- A column with sentiment labels ('positive', 'negative', 'neutral')

### Adjusting Model Parameters

Edit the models in `train_model.py`:

```python
self.models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=200)
}
```

## 📈 Performance

The model's performance depends on the training data. With the sample data:
- Training samples: ~20
- Testing samples: ~5
- Expected accuracy: 60-80% (varies due to small dataset)

For better performance, use a larger dataset like:
- Twitter Sentiment Analysis Dataset
- Amazon Product Reviews
- IMDB Movie Reviews

## 🐛 Troubleshooting

### Issue: Model files not found
**Solution**: Run `python train_model.py` first to train and save the model

### Issue: NLTK data not found
**Solution**: The script will automatically download required NLTK data. If it fails, manually run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue: Streamlit not opening
**Solution**: Check if port 8501 is available. Use a different port:
```bash
streamlit run app.py --server.port 8502
```

## 📦 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models
- **nltk**: Natural language processing
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **matplotlib**: Static plots
- **seaborn**: Statistical visualizations

## 🎓 Lab Task Submission

For the lab task submission:

1. ✅ Train the model: `python train_model.py`
2. ✅ Test the Streamlit app: `streamlit run app.py`
3. ✅ Record a video showing:
   - Model training output
   - Streamlit app functionality
   - Single text prediction
   - Batch CSV analysis
4. ✅ Upload code to GitHub (public repository)
5. ✅ Post video on YouTube
6. ✅ Share links on LinkedIn with Uneeq Interns tagged

## 👨‍💻 Author

Created for Uneeq Internship - Egypt

## 📄 License

This project is open source and available for educational purposes.

---

**Happy Analyzing! 😊**
