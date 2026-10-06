def calculate_ltv(monthly_charges, tenure, churn_probability, horizon_months=12):
    """
    Calculate risk-adjusted projected customer lifetime value.

    This is a formula-based estimate, not a trained LTV regression model.
    """

    monthly_charges = float(monthly_charges)
    tenure = float(tenure)
    churn_probability = float(churn_probability)
    horizon_months = int(horizon_months)

    if monthly_charges < 0:
        raise ValueError("monthly_charges cannot be negative.")

    if tenure < 0:
        raise ValueError("tenure cannot be negative.")

    if not 0 <= churn_probability <= 1:
        raise ValueError(
            "churn_probability must be between 0 and 1."
        )

    if not 1 <= horizon_months <= 120:
        raise ValueError(
            "horizon_months must be between 1 and 120."
        )

    retention_probability = 1 - churn_probability

    estimated_ltv = (
        monthly_charges
        * horizon_months
        * retention_probability
    )

    return round(estimated_ltv, 2)