"""
Prometheus metrics for recommendation API.

Tracks:
- Request count and latency
- Cache hit/miss rates
- Model prediction counts
- Error rates
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time
from functools import wraps

# ============================================================================
# Metrics Definitions
# ============================================================================

# Request metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_latency = Histogram(
    'api_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

# Recommendation metrics
recommendations_count = Counter(
    'recommendations_total',
    'Total recommendations generated',
    ['model']
)

recommendations_latency = Histogram(
    'recommendation_duration_seconds',
    'Time to generate recommendations',
    ['model']
)

# Cache metrics
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Cache hit rate percentage',
    ['cache_type']
)

# Model metrics
model_predictions = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model']
)

prediction_scores = Histogram(
    'prediction_scores',
    'Distribution of prediction scores',
    ['model']
)

# Error metrics
errors_count = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type']
)

# System metrics
active_users = Gauge(
    'active_users',
    'Number of active users in last 5 minutes'
)

models_loaded = Gauge(
    'models_loaded',
    'Number of models loaded'
)


# ============================================================================
# Metric Tracking Functions
# ============================================================================

def track_request(method: str, endpoint: str, status: int):
    """Track API request."""
    request_count.labels(method=method, endpoint=endpoint, status=status).inc()


def track_cache_hit(cache_type: str = "redis"):
    """Track cache hit."""
    cache_hits.labels(cache_type=cache_type).inc()


def track_cache_miss(cache_type: str = "redis"):
    """Track cache miss."""
    cache_misses.labels(cache_type=cache_type).inc()


def update_cache_hit_rate():
    """Calculate and update cache hit rate."""
    for cache_type in ["redis"]:
        hits = cache_hits.labels(cache_type=cache_type)._value._value
        misses = cache_misses.labels(cache_type=cache_type)._value._value
        total = hits + misses
        if total > 0:
            hit_rate = (hits / total) * 100
            cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)


def track_recommendation(model: str, duration: float):
    """Track recommendation generation."""
    recommendations_count.labels(model=model).inc()
    recommendations_latency.labels(model=model).observe(duration)


def track_prediction(model: str, score: float):
    """Track model prediction."""
    model_predictions.labels(model=model).inc()
    prediction_scores.labels(model=model).observe(score)


def track_error(endpoint: str, error_type: str):
    """Track API error."""
    errors_count.labels(endpoint=endpoint, error_type=error_type).inc()


# ============================================================================
# Decorators
# ============================================================================

def track_time(metric_func):
    """Decorator to track execution time."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric_func(duration)
        return wrapper
    return decorator


# ============================================================================
# Metrics Endpoint
# ============================================================================

def metrics_endpoint() -> Response:
    """
    Generate Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format.
    """
    # Update dynamic metrics
    update_cache_hit_rate()

    # Generate metrics
    metrics = generate_latest()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)
