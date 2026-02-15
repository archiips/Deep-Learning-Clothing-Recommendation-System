"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime


class RecommendationRequest(BaseModel):
    """Request schema for getting recommendations."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 100,
                "k": 10,
                "model": "mf",
                "department": None
            }
        }
    )

    user_id: int = Field(..., description="User ID for recommendations", ge=0)
    k: int = Field(10, ge=1, le=20, description="Number of recommendations to return")
    model: str = Field("mf", description="Model to use: 'popularity', 'mf', or 'ncf'")
    department: Optional[str] = Field(None, description="Filter by department (e.g., 'Tops', 'Dresses')")


class RecommendationItem(BaseModel):
    """Individual recommendation item with metadata."""
    clothing_id: int = Field(..., description="Clothing item ID")
    predicted_score: float = Field(..., description="Predicted score/rating")
    rank: int = Field(..., description="Rank in recommendation list (1-based)")
    department: str = Field(..., description="Product department")
    class_name: str = Field(..., description="Product class")
    avg_rating: float = Field(..., description="Average rating from all users")
    num_reviews: int = Field(..., description="Number of reviews")


class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 100,
                "recommendations": [
                    {
                        "clothing_id": 1234,
                        "predicted_score": 4.8,
                        "rank": 1,
                        "department": "Tops",
                        "class_name": "Blouses",
                        "avg_rating": 4.5,
                        "num_reviews": 120
                    }
                ],
                "model_used": "mf",
                "timestamp": "2026-02-15T12:00:00",
                "total_items": 365
            }
        }
    )

    user_id: int = Field(..., description="User ID")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")
    model_used: str = Field(..., description="Model that generated recommendations")
    timestamp: str = Field(..., description="ISO timestamp of request")
    total_items: int = Field(..., description="Total number of items available")


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 100,
                "clothing_id": 1234,
                "model": "mf"
            }
        }
    )

    user_id: int = Field(..., description="User ID", ge=0)
    clothing_id: int = Field(..., description="Clothing item ID", ge=0)
    model: str = Field("mf", description="Model to use: 'popularity', 'mf', or 'ncf'")


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 100,
                "clothing_id": 1234,
                "predicted_rating": 4.75,
                "model_used": "mf",
                "timestamp": "2026-02-15T12:00:00"
            }
        }
    )

    user_id: int
    clothing_id: int
    predicted_rating: float
    model_used: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    num_users: int
    num_items: int
    available_models: List[str]


class ModelInfo(BaseModel):
    """Model information."""
    model_name: str
    type: str
    description: str
    parameters: dict


class ModelsResponse(BaseModel):
    """Response listing all available models."""
    models: List[ModelInfo]
    default_model: str
