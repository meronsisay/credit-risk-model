"""
Pydantic models for request/response validation.
These act as a contract between API and clients.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class CustomerFeatures(BaseModel):
    """
    Input schema for single customer prediction.
    Must match exactly the features your model was trained on.
    """
    
    # Core numerical features
    avg_amount: float = Field(..., ge=0, description="Average transaction amount")
    std_amount: float = Field(..., ge=0, description="Standard deviation of amounts")
    weekend_ratio: float = Field(..., ge=0, le=1, description="Weekend transaction ratio")
    business_hour_ratio: float = Field(..., ge=0, le=1, description="Business hour ratio")
    transaction_hour_std: float = Field(..., ge=0, description="Hour standard deviation")
    unique_productcategory: int = Field(..., ge=1, description="Unique product categories")
    unique_channelid: int = Field(..., ge=1, description="Unique channels")
    unique_providerid: int = Field(..., ge=1, description="Unique providers")
    
    # One-hot encoded features (from your training)
    cat_airtime: Optional[int] = Field(default=0, ge=0, le=1)
    cat_financial_services: Optional[int] = Field(default=0, ge=0, le=1)
    cat_tv: Optional[int] = Field(default=0, ge=0, le=1)
    cat_utility_bill: Optional[int] = Field(default=0, ge=0, le=1)
    cat_data_bundles: Optional[int] = Field(default=0, ge=0, le=1)
    cat_movies: Optional[int] = Field(default=0, ge=0, le=1)
    cat_transport: Optional[int] = Field(default=0, ge=0, le=1)
    cat_ticket: Optional[int] = Field(default=0, ge=0, le=1)
    
    channel_1: Optional[int] = Field(default=0, ge=0, le=1)
    channel_2: Optional[int] = Field(default=0, ge=0, le=1)
    channel_3: Optional[int] = Field(default=0, ge=0, le=1)
    channel_4: Optional[int] = Field(default=0, ge=0, le=1)
    
    provider_1: Optional[int] = Field(default=0, ge=0, le=1)
    provider_2: Optional[int] = Field(default=0, ge=0, le=1)
    provider_3: Optional[int] = Field(default=0, ge=0, le=1)
    provider_4: Optional[int] = Field(default=0, ge=0, le=1)
    provider_5: Optional[int] = Field(default=0, ge=0, le=1)
    provider_6: Optional[int] = Field(default=0, ge=0, le=1)
    
    class Config:
        # Allow extra fields to be ignored (graceful degradation)
        extra = "ignore"


class PredictionResponse(BaseModel):
    """Output schema for prediction."""
    
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of being high-risk")
    risk_label: str = Field(..., description="'high_risk' or 'low_risk'")
    model_version: Optional[str] = Field(None, description="Model version used")


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""
    
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_version: Optional[str]