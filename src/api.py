import io
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic import ValidationError

from src.models.predict import (
    AVAILABLE_MODELS,
    DEFAULT_HORIZON_MONTHS,
    DEFAULT_MODEL_NAME,
    MODEL_DIR,
    load_artifacts,
    score_customers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_CUSTOMERS = 1000

app = FastAPI(
    title="Customer Churn & Revenue Risk API",
    description=(
        "Scores customer churn risk and estimates recurring revenue at risk "
        "over a selected planning horizon."
    ),
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ScoreRequest(BaseModel):
    customers: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_CUSTOMERS,
    )
    horizon_months: int = Field(
        default=DEFAULT_HORIZON_MONTHS,
        ge=1,
        le=120,
    )
    model_name: Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "logistic_regression_tuned",
        "random_forest_tuned",
    ] = DEFAULT_MODEL_NAME


def _result_payload(results: pd.DataFrame) -> dict[str, Any]:
    records = results.to_dict(orient="records")
    high_risk_count = int((results["risk_category"] == "High").sum())
    return {
        "count": len(records),
        "high_risk_count": high_risk_count,
        "total_revenue_at_risk": float(results["revenue_at_risk"].sum()),
        "records": records,
    }


async def _read_limited_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_CSV_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Request body exceeds 10 MB.",
            )
    return bytes(body)


def _score(customers: pd.DataFrame, horizon_months: int, model_name: str):
    try:
        results = score_customers(
            customers,
            horizon_months=horizon_months,
            model_name=model_name,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _result_payload(results)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    artifacts_ready = (
        (MODEL_DIR / "preprocessor.joblib").is_file()
        and (MODEL_DIR / f"{DEFAULT_MODEL_NAME}.joblib").is_file()
    )
    return {
        "status": "ok" if artifacts_ready else "needs_training",
        "model_artifacts_ready": artifacts_ready,
        "default_model": DEFAULT_MODEL_NAME,
    }


@app.get("/api/schema")
def input_schema():
    try:
        _, preprocessor = load_artifacts()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    numerical_columns = preprocessor.transformers_[0][2]
    categorical_columns = preprocessor.transformers_[1][2]
    encoder = preprocessor.named_transformers_["categorical"]
    options_by_column = dict(zip(categorical_columns, encoder.categories_))

    fields = []
    for column in numerical_columns:
        fields.append(
            {
                "name": column,
                "kind": "number",
                "default": 70 if column == "MonthlyCharges" else 0,
            }
        )
    for column in categorical_columns:
        options = [str(value) for value in options_by_column[column]]
        fields.append(
            {
                "name": column,
                "kind": "category",
                "options": options,
                "default": options[0] if options else "",
            }
        )

    available_models = [
        name
        for name in sorted(AVAILABLE_MODELS)
        if (MODEL_DIR / f"{name}.joblib").is_file()
    ]
    default_model = (
        DEFAULT_MODEL_NAME
        if DEFAULT_MODEL_NAME in available_models
        else available_models[0]
    )
    return {
        "fields": fields,
        "models": available_models,
        "default_model": default_model,
        "default_horizon_months": DEFAULT_HORIZON_MONTHS,
        "revenue_estimate_note": (
            "Revenue at risk is churn probability × monthly charges × "
            "planning-horizon months. The horizon does not calibrate the "
            "churn score; this is not a full customer lifetime value forecast."
        ),
    }


@app.post(
    "/api/score",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": ScoreRequest.model_json_schema(),
                }
            },
        }
    },
)
async def score(request: Request):
    body = await _read_limited_body(request)
    try:
        score_request = ScoreRequest.model_validate_json(body)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    customers = pd.DataFrame(score_request.customers)
    return _score(
        customers,
        score_request.horizon_months,
        score_request.model_name,
    )


@app.post("/api/score-csv")
async def score_csv(
    request: Request,
    horizon_months: int = Query(default=DEFAULT_HORIZON_MONTHS, ge=1, le=120),
    model_name: str = Query(default=DEFAULT_MODEL_NAME),
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=422, detail="Unknown model name.")

    body = await _read_limited_body(request)
    if not body:
        raise HTTPException(status_code=422, detail="CSV file is empty.")
    if len(body) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV file exceeds 10 MB.")

    try:
        customers = pd.read_csv(io.BytesIO(body))
    except (
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail="Could not parse the uploaded file as CSV.",
        ) from error
    if len(customers) > MAX_CUSTOMERS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV batch exceeds the {MAX_CUSTOMERS}-customer limit.",
        )
    return _score(customers, horizon_months, model_name)