# 🎯 Customer Churn Prediction System

A complete machine learning application for predicting customer churn in subscription-based services. Built with FastAPI backend and premium Streamlit frontend.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🤖 **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- ⚖️ **Imbalanced Data Handling**: SMOTE implementation for balanced predictions
- 🎯 **High Accuracy**: Focus on precision, recall, and F1-score metrics
- 🚀 **FastAPI Backend**: RESTful API with automatic documentation
- 💎 **Premium UI**: Dark theme with glassmorphism and smooth animations
- 📊 **Interactive Visualizations**: Plotly charts for insights
- 📁 **Batch Processing**: Upload CSV files for multiple predictions
- 📈 **Model Insights**: Performance metrics and feature importance

## 📸 Screenshots

### Single Prediction Interface
Beautiful dark-themed interface with gradient backgrounds and glassmorphism effects.

### Batch Prediction Dashboard
Upload CSV files and get predictions for thousands of customers instantly.

### Model Performance Metrics
Comprehensive insights into model performance and feature importance.

## 🏗️ Project Structure

```
Customer Churn prediction/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed data
├── models/
│   ├── churn_model.pkl        # Trained model
│   ├── preprocessor.pkl       # Data preprocessor
│   ├── model_metrics.json     # Performance metrics
│   └── feature_importance.csv # Feature rankings
├── src/
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   └── train.py               # Model training script
├── backend/
│   ├── main.py                # FastAPI application
│   └── model_handler.py       # Model loading and predictions
├── frontend/
│   ├── app.py                 # Streamlit application
│   └── styles.css             # Custom CSS styling
├── notebooks/
│   └── model_training.ipynb   # Jupyter notebook for EDA
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (or navigate to the project directory)

```bash
cd "e:\Interships\Uneeq intern Egypt\Customer Churn prediction"
```

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate the virtual environment**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Training the Model

Train the machine learning model with multiple algorithms:

```bash
python src/train.py
```

This will:
- Create a synthetic customer churn dataset (if none exists)
- Apply SMOTE for handling class imbalance
- Train 4 different models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Evaluate models using precision, recall, F1-score, and ROC-AUC
- Select and save the best model
- Generate feature importance rankings

**Output Files**:
- `models/churn_model.pkl` - Best performing model
- `models/preprocessor.pkl` - Data preprocessing pipeline
- `models/model_metrics.json` - Performance metrics
- `models/feature_importance.csv` - Feature rankings

### Running the Application

#### 1. Start the FastAPI Backend

```bash
cd backend
python main.py
```

Or using uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

#### 2. Start the Streamlit Frontend

Open a new terminal and run:

```bash
cd frontend
streamlit run app.py
```

The app will open automatically at http://localhost:8501

## 📖 Usage Guide

### Single Customer Prediction

1. Navigate to **🔮 Single Prediction** page
2. Fill in customer information:
   - Demographics (Gender, Senior Citizen, Partner, Dependents)
   - Phone services
   - Internet services
   - Account information (Tenure, Contract, Payment Method, Charges)
3. Click **🎯 Predict Churn**
4. View results with probability, risk level, and recommendations

### Batch Predictions

1. Navigate to **📁 Batch Prediction** page
2. Upload a CSV file with customer data
3. Preview your data
4. Click **🎯 Predict All**
5. View summary statistics and risk distribution
6. Download results as CSV

**Required CSV Columns**:
```
Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, 
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
DeviceProtection, TechSupport, StreamingTV, StreamingMovies, 
Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
```

### Model Insights

1. Navigate to **📈 Model Insights** page
2. View model performance comparison
3. See confusion matrix for the best model
4. Analyze top 10 most important features

## 🔌 API Endpoints

### Health Check
```
GET /
GET /health
```

### Single Prediction
```
POST /predict
Content-Type: application/json

{
  "Gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "Tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 844.20
}
```

### Batch Prediction
```
POST /predict/batch
Content-Type: multipart/form-data

file: customers.csv
```

### Model Information
```
GET /model/info
GET /features/importance
```

## 📊 Model Performance

The system trains and compares multiple classification algorithms:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.81 | ~0.67 | ~0.55 | ~0.60 | ~0.85 |
| Random Forest | ~0.80 | ~0.65 | ~0.52 | ~0.58 | ~0.84 |
| XGBoost | ~0.82 | ~0.68 | ~0.57 | ~0.62 | ~0.86 |
| LightGBM | ~0.82 | ~0.68 | ~0.56 | ~0.61 | ~0.86 |

*Note: Actual metrics may vary based on dataset and SMOTE application*

### Key Features Influencing Churn

1. **Contract Type** - Month-to-month contracts have higher churn
2. **Tenure** - New customers are more likely to churn
3. **Monthly Charges** - Higher charges correlate with churn
4. **Internet Service** - Fiber optic customers show different patterns
5. **Payment Method** - Electronic check users have higher churn

## 🎨 UI Features

- **Dark Theme**: Eye-friendly dark color scheme
- **Glassmorphism**: Modern frosted glass effects
- **Gradient Backgrounds**: Vibrant purple and blue gradients
- **Smooth Animations**: Hover effects and transitions
- **Responsive Design**: Works on all screen sizes
- **Interactive Charts**: Plotly visualizations
- **Professional Typography**: Google Fonts (Inter, Poppins)

## 🛠️ Technologies Used

### Backend
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **Python-multipart**: File upload support

### Frontend
- **Streamlit**: Web application framework
- **Plotly**: Interactive charts
- **Requests**: API communication

### Machine Learning
- **Scikit-learn**: ML algorithms and preprocessing
- **XGBoost**: Gradient boosting
- **LightGBM**: Gradient boosting
- **Imbalanced-learn**: SMOTE implementation

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

## 📝 Dataset

The project includes a synthetic customer churn dataset generator with realistic features:

- **Size**: Configurable (default 3,000 customers)
- **Features**: 19 customer attributes
- **Target**: Binary churn indicator (Yes/No)
- **Class Distribution**: Imbalanced (~27% churn rate)
- **Realism**: Correlated features mimicking real telecom data

You can also use your own dataset by placing it in `data/raw/churn_data.csv`

## 🔐 Security Notes

- For production deployment, update CORS settings in `backend/main.py`
- Use environment variables for sensitive configuration
- Implement authentication for API endpoints
- Add rate limiting for public APIs

## 🚀 Deployment

### Docker Deployment (Optional)

Create a `Dockerfile` for containerization:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

- **Backend**: Deploy FastAPI on platforms like Heroku, AWS, or Google Cloud
- **Frontend**: Deploy Streamlit on Streamlit Cloud or Heroku
- **Model**: Store models in cloud storage (S3, GCS)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Created as part of the Uneeq Internship Project

## 🙏 Acknowledgments

- Dataset inspired by Telco Customer Churn datasets
- UI design inspired by modern web applications
- Machine learning approach follows industry best practices

## 📞 Support

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ using FastAPI, Streamlit, and Machine Learning**
