def calculate_ltv(monthly_charges, tenure, churn_probability):
    """
    Estimate customer lifetime value.
    """

    monthly_charges = float(monthly_charges)
    tenure = float(tenure)
    churn_probability = float(churn_probability)

    # Estimated remaining lifetime in months
    retention_probability = 1 - churn_probability
    remaining_months = max(1, tenure * retention_probability)

    # Estimated lifetime value
    ltv = monthly_charges * remaining_months

    return round(ltv, 2)