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

**Key requirements:** Interpretability, documentation, and reproducibility are mandatory for regulatory compliance. Logistic Regression with WoE is preferred for its transparency.

---

### 2. Proxy Variables: Necessity and Risks

**Why a proxy is needed:** The dataset contains only transaction records — no direct default label exists.

**Our approach:** Create a proxy target using RFM (Recency, Frequency, Monetary) analysis to identify high-risk customers.

| Risk | Mitigation |
|------|-------------|
| Measurement Error | Validate against real data when available |
| Regulatory Risk | Document proxy rationale thoroughly |
| Bias & Fairness | Test for disparate impact |
| Model Drift | Implement monitoring and retraining |

---

### 3. Model Trade-offs in a Regulated Context

| Dimension | Logistic Regression + WoE | Gradient Boosting |
|-----------|--------------------------|-------------------|
| Interpretability | High | Low (requires SHAP/LIME) |
| Performance | Moderate | High |
| Regulatory Acceptance | Strong | Requires extra validation |

**Recommended Strategy:** Start with Logistic Regression + WoE as baseline, benchmark against Gradient Boosting, and adopt complex model only if performance gain >5% with acceptable explainability.

---

## EDA Summary

### Dataset Overview
| Metric | Value |
|--------|-------|
| Total transactions | 95,662 |
| Total customers | 3,742 |
| Time period | 90 days (Nov 15, 2018 - Feb 13, 2019) |
| Missing values | None |

### Key Findings

| # | Finding | Action |
|---|---------|--------|
| 1 | Fraud rate: 0.2% (193 transactions) | Use SMOTE or class weights |
| 2 | Amount: mean 6,718, median 1,000 (highly skewed) | Apply log transform |
| 3 | 40% of transactions are negative (refunds) | Create `refund_rate` feature |
| 4 | Friday volume doubles other days | Create `is_friday` flag |
| 5 | Top 10% customers drive 63% of volume | Cap or log-transform customer aggregates |
| 6 | Amount vs Value: 0.99 correlation | **Drop Value column** |
| 7 | Provider 3: 2.08% fraud rate | Keep ProviderId as feature |
| 8 | Channel 1: 0.74% fraud rate | Keep ChannelId as feature |

---

## Data Processing Pipeline

### Feature Engineering Output

| Metric | Value |
|--------|-------|
| Customers processed | 3,742 |
| Final features | 47 |
| High-risk target (is_high_risk) | 31.80% (1,190 customers) |
| Low-risk (0) | 68.20% (2,552 customers) |

### Proxy Target Creation (RFM + K-Means)

| Cluster | Size | Recency (days) | Frequency | Monetary (UGX) | Risk |
|---------|------|----------------|-----------|----------------|------|
| 0 | 1,208 | 18.3 | 4.5 | 29,054 | Low |
| 1 | 1,344 | 11.5 | 62.9 | 521,308 | Low |
| 2 | 1,190 | 64.3 | 4.8 | 47,951 | **High** |

**High-risk cluster identified:** Cluster 2 (highest recency, lowest frequency)

### Information Value (IV) - Top Features

| Feature | IV | Predictive Power |
|---------|-----|------------------|
| refund_amount | 0.8182 | STRONG |
| unique_providerid | 0.6515 | STRONG |
| refund_rate | 0.6483 | STRONG |
| transaction_hour_std | 0.5774 | STRONG |
| unique_productcategory | 0.5469 | STRONG |

### Leakage Prevention
RFM features (recency, frequency, monetary) used only for target creation, excluded from predictors.

### Output Files

- `data/processed/processed_data.csv` - Model-ready dataset with `is_high_risk` target
- `models/feature_pipeline.pkl` - Fitted pipeline for inference



## Environment Setup

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/meronsisay/credit-risk-model.git
cd credit-risk-model

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

jupyter notebook notebooks/eda.ipynb
