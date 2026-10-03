from fastapi import FastAPI
from pydantic import BaseModel

from src.models.predict import predict_churn


app = FastAPI(
    title="Customer Churn & LTV Engine",
    version="1.0"
)


class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn & LTV Engine API is running"
    }


@app.post("/predict")
def predict(customer: CustomerData):
    return predict_churn(customer.model_dump())