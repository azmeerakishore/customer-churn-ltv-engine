import asyncio

import httpx
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import api
from src.features.feature_engineering import create_preprocessor, prepare_dataset
from src.models.predict import score_customers
from src.preprocessing.preprocess import load_data


def api_request(method, path, **kwargs):
    async def send_request():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send_request())


@pytest.fixture(scope="module")
def model_bundle():
    features, target = prepare_dataset()
    sample_features = features.sample(n=900, random_state=42)
    sample_target = target.loc[sample_features.index]
    preprocessor = create_preprocessor(sample_features)
    transformed = pd.DataFrame(
        preprocessor.fit_transform(sample_features),
        columns=preprocessor.get_feature_names_out(),
        index=sample_features.index,
    )
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(transformed, sample_target)
    return model, preprocessor


def test_score_customers_returns_probability_and_revenue_estimate(model_bundle):
    model, preprocessor = model_bundle
    customer = load_data().head(1)

    result = score_customers(
        customer,
        horizon_months=6,
        model=model,
        preprocessor=preprocessor,
    )

    assert len(result) == 1
    assert 0 <= result.loc[0, "churn_probability"] <= 1
    assert result.loc[0, "risk_category"] in {"Low", "Medium", "High"}
    assert result.loc[0, "revenue_at_risk"] == pytest.approx(
        result.loc[0, "churn_probability"]
        * result.loc[0, "monthly_charges"]
        * 6
    )
    assert result.loc[0, "horizon_months"] == 6


def test_duplicate_customers_remain_separate_results(model_bundle):
    model, preprocessor = model_bundle
    customer = load_data().head(1)
    customers = pd.concat([customer, customer], ignore_index=True)

    result = score_customers(
        customers,
        model=model,
        preprocessor=preprocessor,
    )

    assert len(result) == 2
    assert result["customerID"].tolist() == customer["customerID"].tolist() * 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.drop(columns="Contract"), "missing required columns"),
        (
            lambda row: row.assign(MonthlyCharges="not-a-number"),
            "MonthlyCharges must contain valid numbers",
        ),
        (lambda row: row.assign(tenure=-1), "tenure cannot be negative"),
        (lambda row: row.assign(SeniorCitizen=2), "SeniorCitizen must be 0 or 1"),
    ],
)
def test_invalid_customer_data_is_rejected(model_bundle, mutation, message):
    model, preprocessor = model_bundle
    customer = mutation(load_data().head(1))

    with pytest.raises(ValueError, match=message):
        score_customers(customer, model=model, preprocessor=preprocessor)


def test_score_api_returns_summary_and_customer_scores(model_bundle, monkeypatch):
    model, preprocessor = model_bundle

    def score_with_test_model(customers, horizon_months, model_name):
        return score_customers(
            customers,
            horizon_months=horizon_months,
            model=model,
            preprocessor=preprocessor,
        )

    monkeypatch.setattr(api, "score_customers", score_with_test_model)
    customer = load_data().head(1).drop(columns="Churn").to_dict("records")

    response = api_request(
        "POST",
        "/api/score",
        json={"customers": customer, "horizon_months": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["high_risk_count"] in {0, 1}
    assert payload["records"][0]["horizon_months"] == 3


def test_csv_api_scores_batch(model_bundle, monkeypatch):
    model, preprocessor = model_bundle

    def score_with_test_model(customers, horizon_months, model_name):
        return score_customers(
            customers,
            horizon_months=horizon_months,
            model=model,
            preprocessor=preprocessor,
        )

    monkeypatch.setattr(api, "score_customers", score_with_test_model)

    csv_content = (
        load_data()
        .head(2)
        .drop(columns="Churn")
        .to_csv(index=False)
    )

    response = api_request(
        "POST",
        "/api/score-csv?horizon_months=9",
        files={
            "file": (
                "customers.csv",
                csv_content.encode("utf-8"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2
    assert "records" in data
    assert len(data["records"]) == 2

    for record in data["records"]:
        assert "customerID" in record
        assert "churn_probability" in record
        assert "risk_category" in record
        assert "revenue_at_risk" in record
        assert "estimated_ltv" in record
        assert record["horizon_months"] == 9


def test_empty_csv_is_rejected():
    response = api_request(
        "POST",
        "/api/score-csv",
        content="",
        headers={"Content-Type": "text/csv"},
    )

    assert response.status_code == 422


def test_json_api_rejects_batches_above_customer_limit():
    response = api_request(
        "POST",
        "/api/score",
        json={"customers": [{}] * (api.MAX_CUSTOMERS + 1)},
    )

    assert response.status_code == 422


def test_csv_api_rejects_oversized_request_body():
    response = api_request(
        "POST",
        "/api/score-csv",
        files={
            "file": (
                "large.csv",
                b"x" * (api.MAX_CSV_BYTES + 1),
                "text/csv",
            )
        },
    )

    assert response.status_code == 413


def test_csv_api_rejects_batches_above_customer_limit():
    customer = load_data().head(1).drop(columns="Churn")
    customers = pd.concat(
        [customer] * (api.MAX_CUSTOMERS + 1),
        ignore_index=True,
    )
    csv_content = customers.to_csv(index=False)

    response = api_request(
        "POST",
        "/api/score-csv",
        files={
            "file": (
                "large_batch.csv",
                csv_content.encode("utf-8"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 413


def test_pages_origin_cors_preflight_is_allowed():
    response = api_request(
        "OPTIONS",
        "/api/score",
        headers={
            "Origin": "https://azmeerakishore.github.io",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://azmeerakishore.github.io"
    )


def test_relative_dashboard_assets_are_served_by_fastapi():
    for path in ("/", "/styles.css", "/app.js"):
        response = api_request("GET", path)
        assert response.status_code == 200