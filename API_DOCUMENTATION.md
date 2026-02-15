# Clothing Recommendation API Documentation

## Overview

RESTful API for serving clothing recommendations using three trained models:
- **Popularity Baseline**: Recommends based on item popularity and ratings
- **Matrix Factorization (MF)**: Collaborative filtering with embeddings
- **Neural Collaborative Filtering (NCF)**: Deep learning dual-path architecture

**Base URL**: `http://localhost:8000`
**Interactive Docs**: `http://localhost:8000/docs`

---

## Quick Start

### 1. Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python run_api.py

# Or use uvicorn directly
uvicorn src.api.main:app --reload --port 8000
```

### 2. Docker Deployment

```bash
# Build and start all services (API + Redis + PostgreSQL)
docker-compose up --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f api
```

### 3. Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 100, "k": 10, "model": "mf"}'
```

---

## API Endpoints

### General Endpoints

#### `GET /` - Root
Returns API information.

**Response:**
```json
{
  "name": "Clothing Recommendation API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

#### `GET /health` - Health Check
Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "num_users": 4283,
  "num_items": 365,
  "available_models": ["popularity", "mf", "ncf"]
}
```

#### `GET /models` - List Models
Get information about available models.

**Response:**
```json
{
  "models": [
    {
      "model_name": "popularity",
      "type": "Baseline",
      "description": "Popularity-based recommendations",
      "parameters": {"use_categories": true, "num_items": 365}
    },
    {
      "model_name": "mf",
      "type": "Matrix Factorization",
      "description": "Collaborative filtering with embeddings",
      "parameters": {
        "n_users": 4283,
        "n_items": 365,
        "embedding_dim": 64
      }
    },
    {
      "model_name": "ncf",
      "type": "Neural Collaborative Filtering",
      "description": "Dual-path GMF + MLP architecture",
      "parameters": {
        "n_users": 4283,
        "n_items": 365,
        "gmf_embedding_dim": 64,
        "mlp_embedding_dim": 32
      }
    }
  ],
  "default_model": "mf"
}
```

---

### Recommendation Endpoints

#### `POST /recommend` - Get Recommendations

Generate personalized recommendations for a user.

**Request Body:**
```json
{
  "user_id": 100,
  "k": 10,
  "model": "mf",
  "department": "Tops"  // Optional filter
}
```

**Parameters:**
- `user_id` (int, required): User ID (0-4282)
- `k` (int, optional): Number of recommendations (1-20, default: 10)
- `model` (string, optional): Model to use - "popularity", "mf", or "ncf" (default: "mf")
- `department` (string, optional): Filter by department - "Tops", "Dresses", "Bottoms", "Intimate", "Jackets", "Trend"

**Response:**
```json
{
  "user_id": 100,
  "recommendations": [
    {
      "clothing_id": 1234,
      "predicted_score": 4.75,
      "rank": 1,
      "department": "Tops",
      "class_name": "Blouses",
      "avg_rating": 4.5,
      "num_reviews": 120
    },
    {
      "clothing_id": 5678,
      "predicted_score": 4.68,
      "rank": 2,
      "department": "Tops",
      "class_name": "Knits",
      "avg_rating": 4.3,
      "num_reviews": 95
    }
  ],
  "model_used": "mf",
  "timestamp": "2026-02-15T12:00:00",
  "total_items": 365
}
```

**Example Requests:**

```bash
# Matrix Factorization (default)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "k": 10,
    "model": "mf"
  }'

# Popularity-based
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "k": 5,
    "model": "popularity"
  }'

# With department filter
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "k": 10,
    "model": "ncf",
    "department": "Dresses"
  }'
```

---

#### `POST /predict` - Predict Rating

Predict rating for a specific user-item pair.

**Request Body:**
```json
{
  "user_id": 100,
  "clothing_id": 1234,
  "model": "mf"
}
```

**Parameters:**
- `user_id` (int, required): User ID
- `clothing_id` (int, required): Clothing item ID
- `model` (string, optional): Model to use (default: "mf")

**Response:**
```json
{
  "user_id": 100,
  "clothing_id": 1234,
  "predicted_rating": 4.75,
  "model_used": "mf",
  "timestamp": "2026-02-15T12:00:00"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "clothing_id": 1234,
    "model": "mf"
  }'
```

---

## Business Rules

The API automatically applies the following business logic:

1. **Quality Filtering**: Removes items with:
   - Average rating < 3.0
   - Number of reviews < 3

2. **Diversity Enforcement**:
   - Max 4 items per department
   - Max 2 items per class (e.g., Blouses, Knits)

3. **Department Filtering**: Optional filter to show only items from specific department

4. **Cold Start Handling**: New users automatically use popularity-based recommendations

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (user/item doesn't exist) |
| 422 | Validation Error (invalid request format) |
| 500 | Internal Server Error |

**Example Error Response:**
```json
{
  "detail": "Invalid model 'invalid_model'. Choose from: ['popularity', 'mf', 'ncf']"
}
```

---

## Model Comparison

| Model | Type | Pros | Cons | Use Case |
|-------|------|------|------|----------|
| **Popularity** | Baseline | Fast, no cold start, explainable | Not personalized | New users, fallback |
| **MF** | Collaborative Filtering | Personalized, efficient, good accuracy | Cold start for new users/items | General recommendations |
| **NCF** | Deep Learning | Highest accuracy, captures complex patterns | Slower inference, black box | Premium recommendations |

**Performance Metrics** (from Phase 4 evaluation):
- **MF**: RMSE 0.821, Hit@10 0.456, NDCG@10 0.312
- **NCF**: RMSE 0.798, Hit@10 0.478, NDCG@10 0.328
- **Popularity**: RMSE 1.124, Hit@10 0.234, NDCG@10 0.187

---

## Configuration

Environment variables (`.env` file):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DEFAULT_MODEL=mf

# Model Paths
MODEL_PATH_POPULARITY=checkpoints/popularity/baseline.pkl
MODEL_PATH_MF=checkpoints/mf/mf_best.pt
MODEL_PATH_NCF=checkpoints/ncf/ncf_best.pt

# Data Paths
USER_TO_IDX_PATH=dataset/user_to_idx.pkl
ITEM_TO_IDX_PATH=dataset/item_to_idx.pkl
USER_FEATURES_PATH=dataset/user_features.csv
ITEM_FEATURES_PATH=dataset/item_features.csv
```

---

## Testing

### Unit Tests
```bash
# Run pytest
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Manual Tests
```bash
# Run manual test script
./test_api_manual.sh
```

### Load Testing (Optional)
```bash
# Install wrk
brew install wrk  # macOS

# Load test
wrk -t4 -c100 -d30s --latency \
  -s post_recommend.lua \
  http://localhost:8000/recommend
```

---

## Deployment Options

### 1. Local Development
```bash
python run_api.py
```

### 2. Docker (Recommended)
```bash
docker-compose up -d
```

### 3. Cloud Deployment (GCP Cloud Run)
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/recsys-api

# Deploy
gcloud run deploy recsys-api \
  --image gcr.io/PROJECT_ID/recsys-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

---

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Logs
```bash
# Docker logs
docker-compose logs -f api

# Local logs
tail -f logs/api.log
```

### Metrics (If enabled)
```bash
# Prometheus metrics (if configured)
curl http://localhost:9090/metrics
```

---

## Support & Documentation

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Source Code**: See `src/api/` directory
- **Model Details**: See `CLAUDE.md` and phase completion documents
