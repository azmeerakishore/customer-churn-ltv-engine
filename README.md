# Customer Churn Prediction & Lifetime Value (LTV) Engine
## Key EDA Findings

- Overall churn rate: **26.49%**
- Month-to-month contract churn: **42.71%**
- One-year contract churn: **11.27%**
- Two-year contract churn: **2.84%**
- Customers with 0–12 months tenure have the highest churn: **47.43%**
- Churned customers generally have higher MonthlyCharges.
- Churned customers generally have lower tenure.
- Fiber optic customers show relatively higher churn.
- Electronic check customers show relatively higher churn.
- Customers without OnlineSecurity and TechSupport show higher churn tendency.

### Baseline Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.70% | 65.94% | 56.42% | 60.81% |
| Random Forest | 76.65% | 55.22% | 63.64% | 59.13% |
| XGBoost | 80.27% | 66.00% | 52.94% | 58.75% |

**Best overall baseline:** Logistic Regression based on Accuracy and F1 Score.

**Best churn recall:** Random Forest.

### SHAP Findings

The most influential features included:

1. Contract — Month-to-month
2. Tenure
3. MonthlyCharges
4. OnlineSecurity
5. InternetService
6. TechSupport
7. TotalCharges
8. PaymentMethod
