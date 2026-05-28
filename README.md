# Credit Risk Probability Model for Alternative Data

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

---

## Credit Scoring Business Understanding

### 1. Basel II, Interpretability, and Documentation

The **Basel II Accord** emphasizes accurate risk measurement, transparency, and regulatory accountability in credit risk modeling.

| Pillar | Focus | Modeling Implication |
|--------|-------|---------------------|
| **Pillar 1** | Minimum Capital Requirements | Banks must estimate Probability of Default (PD), LGD, and EAD |
| **Pillar 2** | Supervisory Review | Models must be validated, documented, and auditable |
| **Pillar 3** | Market Discipline | Risk processes must be transparent to stakeholders |

Because lending decisions affect both customers and regulatory capital, banks must understand and explain how predictions are made. This creates a strong need for:

- **Interpretability** — understanding how features influence risk predictions
- **Documentation** — recording assumptions, data sources, validation, and monitoring procedures
- **Reproducibility** — enabling independent audits and validation

Highly interpretable models such as Logistic Regression with Weight of Evidence (WoE) are commonly preferred because they align well with Basel II transparency and governance requirements.

---

### 2. Proxy Variables: Necessity and Risks

Traditional credit scoring relies on observed default behavior, such as missed repayments or loan delinquency. However, this dataset contains only transaction records and fraud indicators, with no direct default label.

To train a supervised machine learning model, a **proxy target variable** must be created. In this project, customer behavior metrics such as **Recency, Frequency, and Monetary (RFM)** patterns are used to estimate credit risk.

While necessary, proxy-based prediction introduces several business risks:

| Risk | Description |
|------|-------------|
| **Measurement Error** | Proxy labels may not perfectly represent actual default behavior |
| **Regulatory Risk** | Regulators may challenge weakly justified proxy definitions |
| **Bias & Fairness** | Behavioral features may unintentionally correlate with protected groups |
| **Model Drift** | Customer behavior patterns may change over time |
| **Business Misalignment** | High engagement does not always imply creditworthiness |

To reduce these risks, proxy definitions should be carefully documented, monitored, and validated against real repayment data when available.

---

### 3. Model Trade-offs in a Regulated Context

| Dimension | Logistic Regression + WoE | Gradient Boosting |
|-----------|--------------------------|-------------------|
| **Interpretability** | High and regulator-friendly | Lower; requires SHAP/LIME explanations |
| **Performance** | Moderate | Typically higher predictive accuracy |
| **Regulatory Acceptance** | Strong industry standard | Requires additional validation |
| **Complexity** | Simple to deploy and maintain | More complex tuning and monitoring |
| **Overfitting Risk** | Lower | Higher without careful tuning |
| **Scorecard Conversion** | Easy | More difficult |

In regulated financial environments, institutions must balance predictive performance with explainability, fairness, and compliance.

For Bati Bank, a practical strategy is to:
1. Use **Logistic Regression + WoE** as a transparent baseline model
2. Compare it with **Gradient Boosting** models
3. Adopt the more complex model only if it provides significant performance gains with acceptable explainability and documentation

This approach balances regulatory safety with predictive performance.