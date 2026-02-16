"""
FastAPI application for clothing recommendation system.
Serves 3 trained models: Popularity Baseline, Matrix Factorization, Neural CF.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any

# Import existing utilities
from utils.model_loader import load_all_models
from utils.logger import get_logger

# Import API components
from src.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelsResponse,
    ModelInfo
)
from src.api.business_rules import apply_business_rules, get_item_metadata
from src.api.cache import get_cache, RecommendationCache
from src.api.metrics import (
    metrics_endpoint,
    track_cache_hit,
    track_cache_miss,
    track_recommendation,
    track_prediction,
    models_loaded as models_loaded_metric
)

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Global state for models and data (loaded on startup)
models: Dict[str, Any] = {}
user_to_idx: Dict[int, int] = {}
item_to_idx: Dict[int, int] = {}
idx_to_item: Dict[int, int] = {}
user_features_df: pd.DataFrame = None
item_features_df: pd.DataFrame = None
cache: RecommendationCache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global models, user_to_idx, item_to_idx, idx_to_item, user_features_df, item_features_df, cache

    logger.info("=" * 60)
    logger.info("Starting Clothing Recommendation API")
    logger.info("=" * 60)

    try:
        # Load models using existing utility
        logger.info("Loading models...")
        pop_model, mf_model, ncf_model = load_all_models()

        models = {
            'popularity': pop_model,
            'mf': mf_model,
            'ncf': ncf_model
        }

        logger.info(f"✅ Loaded {len(models)} models successfully")

        # Update Prometheus metric
        models_loaded_metric.set(len(models))

        # Load mappings
        logger.info("Loading user/item mappings...")
        user_to_idx_path = os.getenv("USER_TO_IDX_PATH", "dataset/user_to_idx.pkl")
        item_to_idx_path = os.getenv("ITEM_TO_IDX_PATH", "dataset/item_to_idx.pkl")

        with open(user_to_idx_path, 'rb') as f:
            user_to_idx = pickle.load(f)
        with open(item_to_idx_path, 'rb') as f:
            item_to_idx = pickle.load(f)

        # Create reverse mapping for items
        idx_to_item = {v: k for k, v in item_to_idx.items()}

        logger.info(f"✅ Loaded {len(user_to_idx)} users, {len(item_to_idx)} items")

        # Load features
        logger.info("Loading feature data...")
        user_features_path = os.getenv("USER_FEATURES_PATH", "dataset/user_features.csv")
        item_features_path = os.getenv("ITEM_FEATURES_PATH", "dataset/item_features.csv")

        user_features_df = pd.read_csv(user_features_path)
        item_features_df = pd.read_csv(item_features_path)

        logger.info(f"✅ Loaded user features: {len(user_features_df)} rows")
        logger.info(f"✅ Loaded item features: {len(item_features_df)} rows")

        # Initialize Redis cache
        logger.info("Initializing cache...")
        cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        cache = get_cache(
            redis_host=redis_host,
            redis_port=redis_port,
            enabled=cache_enabled
        )
        if cache.enabled:
            logger.info(f"✅ Cache initialized: {redis_host}:{redis_port}")
        else:
            logger.warning("⚠️ Cache disabled or unavailable")

        logger.info("=" * 60)
        logger.info("API ready to serve requests!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Failed to load models/data: {e}")
        raise

    yield

    # Cleanup on shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Clothing Recommendation API",
    description="Production-ready recommendation system with 3 model architectures: Popularity Baseline, Matrix Factorization, and Neural Collaborative Filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Clothing Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="healthy",
        models_loaded=len(models) > 0,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        available_models=list(models.keys())
    )


@app.get("/models", response_model=ModelsResponse, tags=["General"])
async def list_models():
    """List all available models with information."""
    model_infos = [
        ModelInfo(
            model_name="popularity",
            type="Baseline",
            description="Popularity-based recommendations using avg_rating * log(num_reviews + 1)",
            parameters={
                "use_categories": True,
                "num_items": len(item_to_idx)
            }
        ),
        ModelInfo(
            model_name="mf",
            type="Matrix Factorization",
            description="Collaborative filtering with user/item embeddings and biases",
            parameters={
                "n_users": models['mf'].n_users,
                "n_items": models['mf'].n_items,
                "embedding_dim": models['mf'].embedding_dim
            }
        ),
        ModelInfo(
            model_name="ncf",
            type="Neural Collaborative Filtering",
            description="Dual-path architecture combining GMF and MLP for recommendations",
            parameters={
                "n_users": models['ncf'].n_users,
                "n_items": models['ncf'].n_items,
                "gmf_embedding_dim": models['ncf'].gmf_embedding_dim,
                "mlp_embedding_dim": models['ncf'].mlp_embedding_dim
            }
        )
    ]

    return ModelsResponse(
        models=model_infos,
        default_model=os.getenv("DEFAULT_MODEL", "mf")
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Prometheus metrics endpoint.

    Exposes metrics for monitoring:
    - Request count and latency
    - Cache hit/miss rates
    - Model prediction counts
    - Error rates

    This endpoint is scraped by Prometheus every 10 seconds.
    """
    return metrics_endpoint()


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations for a user.

    Args:
        request: RecommendationRequest with user_id, k, model, and optional department filter

    Returns:
        RecommendationResponse with ranked recommendations and metadata
    """
    # Validate model choice
    if request.model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Choose from: {list(models.keys())}"
        )

    # Check if user exists in training data
    if request.user_id not in user_to_idx:
        # For new users, use popularity-based recommendations
        logger.warning(f"User {request.user_id} not in training data, using popularity model")
        model_to_use = 'popularity'
        user_idx = 0  # Use arbitrary user for popularity model
    else:
        model_to_use = request.model
        user_idx = user_to_idx[request.user_id]

    try:
        # Check cache first
        cached_recommendations = cache.get(
            user_id=request.user_id,
            model=model_to_use,
            k=request.k,
            department=request.department
        )

        if cached_recommendations is not None:
            logger.info(f"Cache HIT for user {request.user_id}")
            track_cache_hit("redis")
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[RecommendationItem(**item) for item in cached_recommendations],
                model_used=model_to_use,
                timestamp=datetime.utcnow().isoformat(),
                total_items=len(item_to_idx)
            )

        logger.info(f"Cache MISS for user {request.user_id}, generating recommendations")
        track_cache_miss("redis")

        # Track recommendation generation time
        import time
        start_time = time.time()

        # Get model
        model = models[model_to_use]

        # Generate recommendations based on model type
        if model_to_use == 'popularity':
            # Get scores for all items
            all_scores = model.get_scores(user_idx)
            # Get top-K item indices
            top_k_indices = np.argsort(all_scores)[::-1][:request.k * 3]  # Get extra for filtering
            predicted_scores = all_scores[top_k_indices]

        elif model_to_use in ['mf', 'ncf']:
            # Get all item indices
            all_item_indices = np.arange(len(item_to_idx))

            # Predict scores for all items
            model.eval()
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx] * len(all_item_indices))
                item_tensor = torch.LongTensor(all_item_indices)
                predictions = model.forward(user_tensor, item_tensor)
                all_scores = predictions.cpu().numpy()

            # Get top-K item indices
            top_k_indices = np.argsort(all_scores)[::-1][:request.k * 3]  # Get extra for filtering
            predicted_scores = all_scores[top_k_indices]

        else:
            raise HTTPException(status_code=500, detail=f"Unknown model type: {model_to_use}")

        # Apply business rules (quality filter, diversity, department filter)
        filtered_indices = apply_business_rules(
            top_k_indices,
            item_features_df,
            department_filter=request.department,
            enforce_quality=True,
            enforce_item_diversity=True
        )

        # Take only top K after filtering
        final_indices = filtered_indices[:request.k]
        final_scores = all_scores[final_indices]

        # Build response items
        recommendation_items = []
        for rank, (item_idx, score) in enumerate(zip(final_indices, final_scores), start=1):
            metadata = get_item_metadata(item_idx, item_features_df, idx_to_item)

            recommendation_items.append(
                RecommendationItem(
                    clothing_id=metadata['clothing_id'],
                    predicted_score=float(score),
                    rank=rank,
                    department=metadata['department'],
                    class_name=metadata['class_name'],
                    avg_rating=metadata['avg_rating'],
                    num_reviews=metadata['num_reviews']
                )
            )

        # Cache recommendations for future requests
        recommendation_dicts = [item.model_dump() for item in recommendation_items]
        cache.set(
            user_id=request.user_id,
            recommendations=recommendation_dicts,
            model=model_to_use,
            k=request.k,
            department=request.department,
            ttl=3600  # 1 hour TTL
        )

        # Track recommendation metrics
        duration = time.time() - start_time
        track_recommendation(model_to_use, duration)

        # Return response
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendation_items,
            model_used=model_to_use,
            timestamp=datetime.utcnow().isoformat(),
            total_items=len(item_to_idx)
        )

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_rating(request: PredictionRequest):
    """
    Predict rating for a specific user-item pair.

    Args:
        request: PredictionRequest with user_id, clothing_id, and model

    Returns:
        PredictionResponse with predicted rating
    """
    # Validate model choice
    if request.model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Choose from: {list(models.keys())}"
        )

    # Validate user and item
    if request.user_id not in user_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"User {request.user_id} not found in training data"
        )

    if request.clothing_id not in item_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Clothing item {request.clothing_id} not found in catalog"
        )

    try:
        # Get indices
        user_idx = user_to_idx[request.user_id]
        item_idx = item_to_idx[request.clothing_id]

        # Get model
        model = models[request.model]

        # Predict based on model type
        if request.model == 'popularity':
            all_scores = model.get_scores(user_idx)
            predicted_rating = float(all_scores[item_idx])

        elif request.model in ['mf', 'ncf']:
            predicted_rating = model.predict(user_idx, item_idx)

        else:
            raise HTTPException(status_code=500, detail=f"Unknown model type: {request.model}")

        return PredictionResponse(
            user_id=request.user_id,
            clothing_id=request.clothing_id,
            predicted_rating=float(predicted_rating),
            model_used=request.model,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting rating: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")


@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """
    Get cache statistics and performance metrics.

    Returns cache hit rate, number of keys, memory usage, etc.
    """
    if cache is None:
        return {"error": "Cache not initialized"}

    return cache.get_stats()


@app.delete("/cache/invalidate/{user_id}", tags=["Cache"])
async def invalidate_user_cache(user_id: int):
    """
    Invalidate all cached recommendations for a specific user.

    Use when user makes a purchase or preferences change.

    Args:
        user_id: User ID to invalidate cache for

    Returns:
        Number of cache keys deleted
    """
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not available")

    deleted_count = cache.invalidate(user_id)
    return {
        "user_id": user_id,
        "deleted_keys": deleted_count,
        "message": f"Invalidated {deleted_count} cache entries for user {user_id}"
    }


@app.delete("/cache/clear", tags=["Cache"])
async def clear_all_cache():
    """
    Clear all cached recommendations.

    Use for cache warmup or maintenance.
    Requires admin privileges in production.
    """
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not available")

    success = cache.clear_all()
    if success:
        return {"message": "All cache cleared successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear cache")


if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    workers = int(os.getenv("API_WORKERS", 1))

    # Run server
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=True
    )
