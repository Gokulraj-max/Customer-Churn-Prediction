from pathlib import Path
from typing import List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    CustomerData,
    PredictionResponse,
    BatchCustomerInput,
    BatchPredictionResponse
)

# Base directories and model path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-grade Machine Learning inference service for predicting telecom customer churn.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for dashboard or external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline safely
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Run 'python src/train.py' to generate.")


def determine_risk(probability: float):
    """Categorize churn probability into operational risk levels and insights."""
    if probability >= 0.70:
        return (
            "High",
            "Customer is at very high risk of churn. Immediate proactive retention outreach recommended."
        )
    elif probability >= 0.40:
        return (
            "Medium",
            "Customer is at moderate risk of churn. Consider offering targeted contract or discount incentives."
        )
    else:
        return (
            "Low",
            "Customer shows strong retention indicators. Standard relationship management advised."
        )


@app.get("/", tags=["General"])
def home():
    """Service metadata and navigation."""
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Monitoring"])
def health():
    """System health check and model loading verification."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    """Returns details on the deployed model pipeline."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model first."
        )
    classifier_name = model.named_steps["classifier"].__class__.__name__
    return {
        "model_type": classifier_name,
        "pipeline_steps": list(model.named_steps.keys()),
        "selected_metric": "ROC-AUC (0.835) & Recall (0.797)",
        "features": {
            "numerical": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
            "categorical": 15
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerData):
    """
    Predict churn likelihood for a single customer profile.
    Returns binary classification, estimated probability, and operational risk category.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently not available. Run 'src/train.py' to generate model artifacts."
        )

    try:
        # Convert Pydantic model to single-row pandas DataFrame
        customer_dict = customer.model_dump()
        customer_df = pd.DataFrame([customer_dict])

        # Model inference
        prediction = int(model.predict(customer_df)[0])
        probability = float(model.predict_proba(customer_df)[0][1])

        risk_level, message = determine_risk(probability)

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level,
            message=message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchCustomerInput):
    """
    Predict churn likelihood for multiple customer records in a single batch request.
    Ideal for automated daily CRM or analytics pipelines.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently not available. Run 'src/train.py' to generate model artifacts."
        )

    try:
        data_list = [c.model_dump() for c in payload.customers]
        df = pd.DataFrame(data_list)

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        results = []
        for pred, prob in zip(predictions, probabilities):
            prob_float = float(prob)
            risk_level, message = determine_risk(prob_float)
            results.append(
                PredictionResponse(
                    churn_prediction=int(pred),
                    churn_probability=round(prob_float, 4),
                    risk_level=risk_level,
                    message=message
                )
            )

        return BatchPredictionResponse(
            total_customers=len(results),
            predictions=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )
