# Data Cleaning Documentation

## Project

Customer Churn Prediction & LTV Engine

## Dataset Summary

* Total Records: 7,043
* Total Columns: 25 (processed dataset)
* Missing Values: 0
* Duplicate Rows: 0

## Cleaning Steps

### 1. Missing Values

* Identified 11 blank values in TotalCharges.
* Converted TotalCharges to numeric format.
* Replaced missing TotalCharges values with 0.
* The affected records had zero tenure.

### 2. Duplicate Records

* Checked for duplicate records.
* No duplicate records were found in the processed dataset.

### 3. Data Types

* Converted TotalCharges to numeric.
* Verified numerical and categorical data types.

### 4. Outlier Check

* Used the IQR method for tenure, MonthlyCharges, and TotalCharges.
* No outliers were detected.

### 5. Data Consistency

* Checked categorical values for consistency.
* Verified numerical ranges and SeniorCitizen values.
* No invalid values were detected in the checks performed.

## Validation Results

* Missing Values: 0
* Duplicate Rows: 0
* Validation Status: PASSED

## Final Output

The cleaned dataset was saved as:

`data/processed/cleaned_dataset.csv`
