import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture
def high_risk_customer_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.8,
        "TotalCharges": 95.8
    }


@pytest.fixture
def low_risk_customer_payload():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.2,
        "TotalCharges": 4433.6
    }


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Customer Churn Prediction API"
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "pipeline_steps" in data


def test_predict_endpoint_high_risk(high_risk_customer_payload):
    response = client.post("/predict", json=high_risk_customer_payload)
    assert response.status_code == 200
    result = response.json()

    assert result["churn_prediction"] in [0, 1]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert "message" in result
    # Typically month-to-month, 1 month tenure, fiber optic is high churn risk
    assert result["churn_probability"] > 0.50


def test_predict_endpoint_low_risk(low_risk_customer_payload):
    response = client.post("/predict", json=low_risk_customer_payload)
    assert response.status_code == 200
    result = response.json()

    assert result["churn_prediction"] in [0, 1]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in ["Low", "Medium", "High"]
    # Long tenure, two-year contract has low churn risk
    assert result["churn_probability"] < 0.40


def test_predict_validation_error_missing_field():
    # Payload missing 'MonthlyCharges' and 'Contract'
    invalid_payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "tenure": 12
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Pydantic Unprocessable Entity


def test_predict_validation_error_negative_tenure(high_risk_customer_payload):
    payload = high_risk_customer_payload.copy()
    payload["tenure"] = -5  # Invalid constraint
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_prediction_endpoint(high_risk_customer_payload, low_risk_customer_payload):
    batch_payload = {
        "customers": [high_risk_customer_payload, low_risk_customer_payload]
    }
    response = client.post("/predict-batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["risk_level"] in ["Medium", "High"]
    assert data["predictions"][1]["risk_level"] == "Low"
