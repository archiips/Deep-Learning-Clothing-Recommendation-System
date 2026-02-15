# Deployment Guide

Complete guide for deploying the Clothing Recommendation API to production.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [GCP Cloud Run Deployment](#gcp-cloud-run-deployment)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Redis Caching](#redis-caching)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Rollback Procedures](#rollback-procedures)
8. [Cost Optimization](#cost-optimization)

---

## Local Development

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/clothing-recommendation-system.git
cd clothing-recommendation-system

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Run API locally
python run_api.py
```

API will be available at: http://localhost:8000

### Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Manual testing
./test_api_manual.sh
```

---

## Docker Deployment

### Local Docker Testing

```bash
# Build Docker image
docker build -t clothing-recsys-api:latest .

# Run container
docker run -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e CACHE_ENABLED=false \
  clothing-recsys-api:latest

# Test
curl http://localhost:8000/health
```

### Docker Compose (Full Stack)

```bash
# Start all services (API + Redis + PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

**Services:**
- API: http://localhost:8000
- Redis: localhost:6379
- PostgreSQL: localhost:5432

---

## GCP Cloud Run Deployment

### Prerequisites

1. **Google Cloud Account**
   - Create account at https://cloud.google.com
   - Enable billing
   - Create new project

2. **Install gcloud CLI**
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Linux
   curl https://sdk.cloud.google.com | bash

   # Authenticate
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Enable APIs**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

### Automated Deployment

```bash
# Set environment variables
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"

# Run deployment script
./deployment/deploy_gcp.sh
```

### Manual Deployment

```bash
# 1. Build and push Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/clothing-recsys-api:latest .
docker push gcr.io/YOUR_PROJECT_ID/clothing-recsys-api:latest

# 2. Deploy to Cloud Run
gcloud run deploy clothing-recsys-api \
  --image gcr.io/YOUR_PROJECT_ID/clothing-recsys-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60s \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars "ENVIRONMENT=production,CACHE_ENABLED=true"

# 3. Get service URL
gcloud run services describe clothing-recsys-api \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

### Verify Deployment

```bash
SERVICE_URL="https://your-service-url.run.app"

# Health check
curl $SERVICE_URL/health

# Test recommendation endpoint
curl -X POST $SERVICE_URL/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 100, "k": 10, "model": "mf"}'
```

---

## CI/CD Pipeline

### GitHub Actions Setup

1. **Add Secrets to GitHub Repository**
   - Settings → Secrets and variables → Actions → New repository secret

   Required secrets:
   ```
   GCP_PROJECT_ID=your-project-id
   GCP_SA_KEY=<service-account-json-key>
   DOCKER_USERNAME=your-dockerhub-username
   DOCKER_PASSWORD=your-dockerhub-password
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   ```

2. **Create GCP Service Account**
   ```bash
   # Create service account
   gcloud iam service-accounts create github-actions \
     --display-name="GitHub Actions"

   # Grant Cloud Run Admin role
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/run.admin"

   # Grant Storage Admin role (for GCR)
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"

   # Create and download key
   gcloud iam service-accounts keys create key.json \
     --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com

   # Copy contents of key.json to GCP_SA_KEY secret
   cat key.json
   ```

3. **Workflow Triggers**
   - **ci-cd.yml**: Runs on push to main/develop, pull requests
   - **model-retrain.yml**: Runs weekly (Sunday 3 AM UTC) or manual trigger

### Manual Workflow Trigger

```bash
# Trigger retraining workflow
gh workflow run model-retrain.yml -f model_type=mf
```

---

## Redis Caching

### Configuration

**Environment Variables:**
```bash
CACHE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_TTL=3600  # 1 hour
```

### Local Redis Setup

```bash
# Install Redis
brew install redis  # macOS
sudo apt install redis-server  # Ubuntu

# Start Redis
redis-server

# Test connection
redis-cli ping  # Should return PONG
```

### Cloud Redis (GCP Memorystore)

```bash
# Create Redis instance
gcloud redis instances create recsys-cache \
  --size=1 \
  --region=us-central1 \
  --tier=basic

# Get connection info
gcloud redis instances describe recsys-cache \
  --region=us-central1 \
  --format='value(host)'
```

### Cache Management

**API Endpoints:**
```bash
# Get cache statistics
curl http://localhost:8000/cache/stats

# Invalidate user cache
curl -X DELETE http://localhost:8000/cache/invalidate/100

# Clear all cache
curl -X DELETE http://localhost:8000/cache/clear
```

**Expected Performance:**
- Cache hit rate target: 80%+
- Latency reduction: 50-200ms → <10ms
- TTL: 1 hour (recommendations)

---

## Monitoring & Maintenance

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "models_loaded": true,
  "num_users": 4283,
  "num_items": 365,
  "available_models": ["popularity", "mf", "ncf"]
}
```

### Logs

**Cloud Run Logs:**
```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=clothing-recsys-api" \
  --limit 50 \
  --format json

# Stream logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=clothing-recsys-api"
```

**Docker Logs:**
```bash
docker-compose logs -f api
```

### Metrics to Monitor

1. **API Performance**
   - Request rate (req/s)
   - Latency (P50, P95, P99)
   - Error rate (%)

2. **Business Metrics**
   - CTR (click-through rate)
   - Conversion rate
   - AOV (average order value)

3. **Model Health**
   - Prediction distribution
   - Coverage
   - Diversity

---

## Rollback Procedures

### Automatic Rollback (Cloud Run)

```bash
# Run rollback script
./deployment/rollback_gcp.sh

# Or manually:
gcloud run services update-traffic clothing-recsys-api \
  --platform managed \
  --region us-central1 \
  --to-revisions PREVIOUS_REVISION=100
```

### Emergency Procedures

**If API is down:**
1. Check Cloud Run logs
2. Verify health endpoint
3. Rollback to previous revision
4. Escalate if issue persists

**If model performance degrades:**
1. Check model metrics (Precision@10, NDCG)
2. Compare to baseline
3. Rollback model checkpoint
4. Trigger manual retraining

---

## Cost Optimization

### Cloud Run Pricing

**Free Tier (per month):**
- 2M requests
- 360,000 GB-seconds
- 180,000 vCPU-seconds

**After Free Tier:**
- Requests: $0.40 per million
- Memory: $0.0000025 per GB-second
- CPU: $0.00001 per vCPU-second

**Estimated Costs:**
- 1M requests/month: ~$6/month
- 10M requests/month: ~$50/month

### Cost Reduction Tips

1. **Set min-instances=0**: No cost when idle
2. **Use caching**: Reduce compute time by 80%
3. **Optimize Docker image**: Smaller images = faster cold starts
4. **Use request-based autoscaling**: Only pay for what you use
5. **Monitor and optimize**: Use Cloud Monitoring to identify waste

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | API host address |
| `API_PORT` | 8000 | API port |
| `API_WORKERS` | 4 | Number of Uvicorn workers |
| `ENVIRONMENT` | development | Environment (development/production) |
| `DEFAULT_MODEL` | mf | Default model (mf/ncf/popularity) |
| `CACHE_ENABLED` | true | Enable Redis caching |
| `REDIS_HOST` | localhost | Redis hostname |
| `REDIS_PORT` | 6379 | Redis port |
| `REDIS_DB` | 0 | Redis database number |
| `REDIS_TTL` | 3600 | Cache TTL in seconds |
| `LOG_LEVEL` | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |

---

## Troubleshooting

### Common Issues

**1. Docker build fails**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t clothing-recsys-api:latest .
```

**2. Models not loading**
```bash
# Check model files exist
ls -lh checkpoints/mf/
ls -lh checkpoints/ncf/
ls -lh checkpoints/popularity/

# Check permissions
chmod 644 checkpoints/*/
```

**3. Redis connection failed**
```bash
# Check Redis is running
redis-cli ping

# Check Redis logs
redis-cli MONITOR

# Disable caching temporarily
export CACHE_ENABLED=false
```

**4. Cloud Run deployment timeout**
```bash
# Increase timeout
gcloud run services update clothing-recsys-api \
  --timeout 300s \
  --region us-central1
```

---

## Support

For issues and questions:
- **Documentation**: See `API_DOCUMENTATION.md`
- **GitHub Issues**: https://github.com/yourusername/clothing-recommendation-system/issues
- **Email**: your-email@example.com

---

## Next Steps

After successful deployment:
1. ✅ Set up monitoring dashboards (Grafana)
2. ✅ Configure alerting rules
3. ✅ Implement model retraining pipeline
4. ✅ Set up A/B testing framework
5. ✅ Add authentication/authorization
6. ✅ Implement rate limiting
7. ✅ Set up custom domain
8. ✅ Configure CDN (Cloud CDN)
