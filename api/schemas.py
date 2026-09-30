from typing import List, Optional
from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Input payload representing customer attributes for churn risk evaluation."""
    gender: str = Field(..., description="Customer gender: 'Female' or 'Male'")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen: 1 or 0")
    Partner: str = Field(..., description="Whether customer has partner: 'Yes' or 'No'")
    Dependents: str = Field(..., description="Whether customer has dependents: 'Yes' or 'No'")
    tenure: int = Field(..., ge=0, description="Number of months customer has stayed with company")
    PhoneService: str = Field(..., description="Whether customer has phone service: 'Yes' or 'No'")
    MultipleLines: str = Field(..., description="'Yes', 'No', or 'No phone service'")
    InternetService: str = Field(..., description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    OnlineBackup: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    TechSupport: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    Contract: str = Field(..., description="'Month-to-month', 'One year', or 'Two year'")
    PaperlessBilling: str = Field(..., description="Whether customer has paperless billing: 'Yes' or 'No'")
    PaymentMethod: str = Field(
        ...,
        description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'"
    )
    MonthlyCharges: float = Field(..., ge=0.0, description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., ge=0.0, description="The total amount charged to the customer")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 256.5
            }
        }
    }


class BatchCustomerInput(BaseModel):
    customers: List[CustomerData]


class PredictionResponse(BaseModel):
    """Model output payload containing binary prediction, probability, risk level, and guidance."""
    churn_prediction: int = Field(..., description="Predicted churn: 1 (churn), 0 (retention)")
    churn_probability: float = Field(..., description="Estimated churn probability between 0.0 and 1.0")
    risk_level: str = Field(..., description="Risk category: 'Low', 'Medium', or 'High'")
    message: str = Field(..., description="Actionable business insight based on prediction")


class BatchPredictionResponse(BaseModel):
    total_customers: int
    predictions: List[PredictionResponse]
