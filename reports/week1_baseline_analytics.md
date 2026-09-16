
# Week 1 Baseline Analytics Report

## 1. Dataset Overview
- Dataset: Telco Customer Churn Dataset
- Total Records: 7,043
- Original Columns: 21
- Target Variable: Churn

## 2. Data Cleaning
- Converted TotalCharges from string to numeric.
- Filled 11 blank TotalCharges values with 0.
- Excluded customerID from model features.
- Removed target columns to prevent data leakage.
- Applied One-Hot Encoding to categorical features.
- Final encoded features: 46.

## 3. Key EDA Findings
- Overall churn rate: approximately 26.54%.
- Month-to-month customers have higher churn.
- Customers with lower tenure show higher churn.
- Higher monthly charges are associated with increased churn.
- Long-term contracts show lower churn rates.

## 4. Baseline Model Results
| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 80.70% | 65.94% | 56.42% | 60.81% |
| Random Forest | 76.65% | 55.22% | 63.64% | 59.13% |
| XGBoost | 80.27% | 66.00% | 52.94% | 58.75% |

## 5. Conclusion
The initial analysis identified important churn-related patterns involving
contract type, tenure, and monthly charges. Baseline models were trained
to support future feature engineering and model improvement.