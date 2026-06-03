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

### Leakage Prevention
RFM features (recency, frequency, monetary) used only for target creation, excluded from predictors.

## Model Performance

| Model | Recall | ROC-AUC | F1 | Precision | Accuracy |
|-------|--------|---------|-----|-----------|----------|
| **Gradient Boosting (Baseline)** | **0.6022** | 0.8160 | 0.6039 | 0.6056 | 0.7489 |
| Logistic Regression (WoE) | 0.5462 | 0.7874 | 0.5516 | 0.5571 | 0.7177 |
| Random Forest (Tuned) | 0.5014 | 0.8139 | 0.5491 | 0.6068 | 0.7382 |

**Selected Model:** Gradient Boosting Baseline (Recall 60% → catches 60 of 100 high-risk customers)

**Registered in MLflow:** `CreditRiskModel` (Version 3)

## API Deployment 

### FastAPI Service

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Single customer risk prediction |
| `/predict/batch` | POST | Batch predictions |


## Output Files

- `data/processed/processed_data.csv` - Model-ready dataset
- `models/feature_pipeline.pkl` - Feature engineering pipeline
- `models/best_model.pkl` - Best trained model
- `models/mlflow.db` - MLflow tracking database

## Project Structure

```
credit-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline (lint + test)
├── data/                 (git)
│   ├── raw/
│   │   └── data.csv              # Raw transactions
│   └── processed/
│       └── processed_data.csv    # Model-ready data with target
├── models/
│   ├── best_model.pkl             # Best trained model
│   ├── feature_pipeline.pkl       # Fitted feature pipeline
│   ├── mlflow.db                  # MLflow SQLite database
│   └── mlruns/                    # MLflow artifacts
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI application
│   │   └── pydantic_models.py     # Request/response validation
│   ├── data_processing.py         # Feature engineering + target creation
│   └── train.py                   # Model training + MLflow tracking
├── tests/
│   ├── test_data_processing.py    # Unit tests
├── notebooks/
│   └── eda.ipynb                  # Exploratory analysis
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt
├── .gitignore
└── README.md

```


## Environment Setup

### Prerequisites
- Python 3.11+
- Git
- Docker Desktop

### Installation & Setup

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
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw data/processed 

# Download data (if not present)
# Place data.csv in data/raw/

# Run feature engineering 
python src/data_processing.py

# Run model training 
python src/train.py

# Launch MLflow UI to view results
mlflow ui --backend-store-uri sqlite:///models/mlflow.db

# Run FastAPI Locally
uvicorn src.api.main:app --reload

# Test the API:

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "avg_amount": 1500,
    "std_amount": 500,
    "weekend_ratio": 0.3,
    "business_hour_ratio": 0.7,
    "transaction_hour_std": 2.5,
    "unique_productcategory": 3,
    "unique_channelid": 2,
    "unique_providerid": 2,
    "cat_airtime": 1
  }'

# View interactive API documentation
# Open http://localhost:8000/docs in your browser

# Build Docker image
docker build -t credit-risk-api .

# Run with Docker
docker run -p 8000:8000 credit-risk-api

# Or use Docker Compose (API + MLflow UI)
docker-compose up --build

# Run Tests & Linter
# Run unit tests
pytest tests/ -v

# Run linter
flake8 src/ tests/ --max-line-length=120
