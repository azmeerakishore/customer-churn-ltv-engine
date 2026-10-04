# Customer Churn Prediction & LTV Engine

An end-to-end demonstration project for preparing the Telco Customer Churn data, training churn classifiers, scoring individual customers or CSV portfolios, and estimating recurring revenue at risk through a small FastAPI application.

## What It Does

- Cleans the source data and prepares a stratified train/test split.
- Trains Logistic Regression, Random Forest, and XGBoost classifiers.
- Evaluates and tunes models, and produces SHAP feature-importance reports.
- Scores customer profiles through a JSON API or the web dashboard.
- Accepts a CSV batch using the same raw feature columns as the source dataset.
- Assigns Low, Medium, or High risk bands at probability boundaries of 1/3 and 2/3.
- Estimates horizon-based recurring revenue at risk.

The data contains a current churn label, not observed customer lifetime or future-value outcomes. Accordingly, the application does not present a statistically trained customer lifetime value (CLV) forecast. Its financial estimate is `churn probability × monthly charges × planning-horizon months`; the horizon scales the revenue scenario and does not change or calibrate the churn probability. Use this as an explainable revenue-at-risk proxy, not a customer value guarantee.

## Quick Start

Use Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m src.models.train
uvicorn src.api:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the dashboard and [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

Training writes model files under `models/`. They are gitignored because they can be regenerated from the included source dataset. The Docker image also trains the baseline models during its build, so a clean Git checkout does not depend on local model files.

## Using the Dashboard

1. Select an installed model and planning horizon.
2. Enter a customer profile and choose **Score customer**, or select a CSV with the source dataset's feature columns.
3. Review churn probability, risk category, monthly charges, and estimated revenue at risk.

The raw CSV in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` is available for a portfolio demonstration. The `Churn` target column is ignored during scoring; `customerID` is optional. Scoring request bodies are limited to 10 MB and batches to 1,000 customers.

## API

- `GET /health` reports whether the default model artifacts are present.
- `GET /api/schema` describes required fields, category options, installed models, and estimate assumptions.
- `POST /api/score` scores one or more JSON customer records.
- `POST /api/score-csv?horizon_months=12` accepts a raw CSV request body.
- `GET /docs` provides interactive Swagger documentation.

The preprocessor is fitted only on the training split and reused for inference. Unknown categorical values are handled by the fitted one-hot encoder; required columns and numeric values are validated before scoring. Risk probability is model output and has not been calibrated for a specific time horizon.

## Project Structure

```text
data/       source and cleaned datasets
models/     generated model and preprocessor artifacts (gitignored)
notebooks/  exploratory data analysis
reports/    evaluation, tuning, cleaning, and SHAP outputs
src/        data, feature, training, inference, and API code
tests/      inference validation and API tests
web/        dashboard HTML, CSS, and JavaScript
```

## Models And Reports

Run the established analysis modules from the repository root:

```bash
python -m src.preprocessing.preprocess
python -m src.features.feature_engineering
python -m src.models.train
python -m src.models.evaluate
python -m src.models.tune
python -m src.models.explain
```

Evaluation results are saved to `reports/model_comparison.csv` and `reports/model_evaluation.json`. Tuned-model results and SHAP output are stored alongside them. Metrics should be read from these generated reports; the app does not claim a performance figure separate from the checked-in evaluation output.

## Tests

```bash
python -m pytest -q
```

The tests build a small model from the included data in memory, so they do not require pre-generated `.joblib` files. GitHub Actions runs this suite on pushes and pull requests.

## Container

```bash
docker build -t customer-churn-ltv-engine .
docker run --rm -p 8000:8000 customer-churn-ltv-engine
```

The container installs the declared dependencies, trains the baseline models from the included dataset, and serves the dashboard/API on port 8000 (or the `PORT` environment variable). This repository contains deployment configuration only; it has not been deployed to a hosting provider.

### Deploying To Render

The repository includes a Render Blueprint in `render.yaml`. To create the service, sign in to Render, choose **New + → Blueprint**, connect this GitHub repository, and deploy the `kishore-development` branch. Render builds the Docker image, trains the baseline artifacts during the image build, and checks `/health` after startup. The first build can take several minutes. A successful configuration does not mean the service is deployed; verify the service status and open the Render-provided URL after deployment.

### Deploying The Dashboard To GitHub Pages

The workflow in `.github/workflows/pages.yml` publishes only `web/` when changes reach `main`. In the repository's **Settings → Pages**, set the build and deployment source to **GitHub Actions**. Merge the dashboard changes into `main` to trigger the workflow; the expected project URL is `https://azmeerakishore.github.io/customer-churn-ltv-engine/`. The static app calls the Render URL configured in `web/app.js`, and the API allows the GitHub Pages origin through CORS. If Render assigns a different service URL, update `API_BASE_URL` in `web/app.js` and the docs link in `web/index.html` to match before publishing.

## Limitations

- The source dataset is a static, historical telco sample; model scores should not be treated as production decisions without external validation.
- The three risk bands are simple display thresholds, not business-optimized intervention cutoffs.
- Revenue at risk is a planning scenario, not true CLV. A true lifetime-value model needs an observed future value or time-to-event target and a documented forecasting horizon.
- The demo API has no authentication, persistence, or customer-data access controls. Do not expose it publicly with real customer data without adding those controls.
