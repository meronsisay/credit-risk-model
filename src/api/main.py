"""
FastAPI application for credit risk prediction.
Serves the trained model via REST API.
"""

import os
import sys
import logging
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.api.pydantic_models import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and metadata
model = None
model_version = None
feature_columns = None

def load_model_from_registry():
    """Load the best model from MLflow registry."""
    global model, model_version, feature_columns
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Point directly to project_root/models/
        model_dir = os.path.normpath(os.path.join(project_root, "models"))
        db_path = os.path.normpath(os.path.join(model_dir, "mlflow.db"))
        
        logger.info(f"Connecting to MLflow Tracking Database at: sqlite:///{db_path}")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        # Get the latest production model
        client = MlflowClient()
        try:
            latest_versions = client.get_latest_versions("CreditRiskModel", stages=["Production"])
            if latest_versions:
                version = latest_versions[0]
                model_version = f"Registry Version {version.version}"
                model_uri = f"models:/CreditRiskModel/{version.version}"
                logger.info(f"Loading model from registry: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
            else:
                raise Exception("No production model found in MLflow registry.")
        except Exception as e:
            logger.warning(f"Registry load failed ({e}), falling back to local file.")
            # Fallback to local model
            model_path = os.path.join(model_dir, "best_model.pkl")
            model = joblib.load(model_path)
            model_version = "local_fallback"
        
        # Get feature columns from model
        if hasattr(model, "feature_names_in_"):
            feature_columns = model.feature_names_in_.tolist()
        else:
            feature_columns = [
                'avg_amount', 'std_amount', 'weekend_ratio', 'business_hour_ratio',
                'transaction_hour_std', 'unique_productcategory', 'unique_channelid',
                'unique_providerid', 'cat_airtime', 'cat_financial_services', 'cat_tv',
                'cat_utility_bill', 'cat_data_bundles', 'cat_movies', 'cat_transport',
                'cat_ticket', 'channel_1', 'channel_2', 'channel_3', 'channel_4',
                'provider_1', 'provider_2', 'provider_3', 'provider_4', 'provider_5', 'provider_6'
            ]
        
        logger.info(f"Model loaded successfully. Version source: {model_version}")
        logger.info(f"Expected features: {len(feature_columns)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model framework completely: {e}")
        return False
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting Credit Risk API...")
    success = load_model_from_registry()
    if not success:
        logger.warning("Model not loaded. API will still run but predictions will fail.")
    yield
    # Shutdown
    logger.info("Shutting down Credit Risk API...")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts probability of customer being high-risk for loan approval",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint to verify API and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=model_version
    )


@app.get("/model/info", tags=["Monitoring"])
async def model_info():
    """Get model information including expected features."""
    return {
        "model_version": model_version,
        "model_loaded": model is not None,
        "expected_features": feature_columns,
        "feature_count": len(feature_columns) if feature_columns else 0
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerFeatures):
    """Predict risk probability for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = customer.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Ensure all expected columns exist (fill missing with 0)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select only the columns the model expects, in correct order
        input_df = input_df[feature_columns]
        
        # Make prediction
        risk_prob = model.predict_proba(input_df)[0, 1]
        risk_label = "high_risk" if risk_prob > 0.5 else "low_risk"
        
        return PredictionResponse(
            risk_probability=round(risk_prob, 4),
            risk_label=risk_label,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict risk probability for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        for customer in request.customers:
            input_dict = customer.model_dump()
            input_df = pd.DataFrame([input_dict])
            
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[feature_columns]
            risk_prob = model.predict_proba(input_df)[0, 1]
            risk_label = "high_risk" if risk_prob > 0.5 else "low_risk"
            
            predictions.append(PredictionResponse(
                risk_probability=round(risk_prob, 4),
                risk_label=risk_label,
                model_version=model_version
            ))
        
        return BatchPredictionResponse(predictions=predictions)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Credit Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict (POST)",
            "predict_batch": "/predict/batch (POST)",
            "docs": "/docs"
        }
    }


# For direct execution (python src/api/main.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)