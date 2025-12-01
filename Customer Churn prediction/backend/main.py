"""
FastAPI Backend for Customer Churn Prediction
Provides REST API endpoints for model predictions
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import io
import uvicorn

from model_handler import get_model_handler


# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class CustomerData(BaseModel):
    """Customer data for churn prediction"""
    Gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    Tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., gt=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=844.20)
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    churn: bool
    churn_probability: float
    retention_probability: float
    risk_level: str
    prediction_label: str
    message: Optional[str] = None


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        handler = get_model_handler()
        return {
            "status": "healthy",
            "model_loaded": handler.model is not None,
            "preprocessor_loaded": handler.preprocessor is not None
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict churn for a single customer
    
    - **customer**: Customer data with all required features
    - Returns prediction with probability and risk level
    """
    try:
        handler = get_model_handler()
        customer_dict = customer.dict()
        
        result = handler.predict_single(customer_dict)
        result['message'] = f"Prediction successful. Customer is predicted to {result['prediction_label']}."
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict churn for multiple customers from CSV file
    
    - **file**: CSV file with customer data
    - Returns CSV file with predictions
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = CustomerData.__fields__.keys()
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        # Make predictions
        handler = get_model_handler()
        results_df = handler.predict_batch(df)
        
        # Convert to CSV
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{file.filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """
    Get model information and performance metrics
    
    Returns model details, metrics, and feature importance
    """
    try:
        handler = get_model_handler()
        info = handler.get_model_info()
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/features/importance")
async def get_feature_importance():
    """Get feature importance for the model"""
    try:
        handler = get_model_handler()
        if handler.feature_importance is not None:
            return {
                "features": handler.feature_importance.to_dict('records')
            }
        else:
            return {"message": "Feature importance not available"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Customer Churn Prediction API Server")
    print("="*60)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
