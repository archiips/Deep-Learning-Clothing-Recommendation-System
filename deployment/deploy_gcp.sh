#!/bin/bash

###############################################################################
# GCP Cloud Run Deployment Script
#
# Deploys the Clothing Recommendation API to Google Cloud Run
#
# Prerequisites:
# 1. Google Cloud SDK installed (gcloud CLI)
# 2. Authenticated: gcloud auth login
# 3. Project configured: gcloud config set project YOUR_PROJECT_ID
# 4. Billing enabled on GCP project
# 5. Cloud Run API enabled: gcloud services enable run.googleapis.com
###############################################################################

set -e  # Exit on error

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="clothing-recsys-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MEMORY="2Gi"
CPU="2"
TIMEOUT="60s"
MIN_INSTANCES=0
MAX_INSTANCES=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}GCP Cloud Run Deployment${NC}"
echo -e "${GREEN}================================================${NC}"

# Step 1: Verify prerequisites
echo -e "\n${YELLOW}Step 1: Verifying prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Install from: https://www.docker.com/get-started${NC}"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${RED}❌ Not authenticated to GCP. Run: gcloud auth login${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites verified${NC}"

# Step 2: Set project
echo -e "\n${YELLOW}Step 2: Setting GCP project...${NC}"
gcloud config set project ${PROJECT_ID}
echo -e "${GREEN}✅ Project set to: ${PROJECT_ID}${NC}"

# Step 3: Enable required APIs
echo -e "\n${YELLOW}Step 3: Enabling required GCP APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
echo -e "${GREEN}✅ APIs enabled${NC}"

# Step 4: Configure Docker for GCR
echo -e "\n${YELLOW}Step 4: Configuring Docker for Google Container Registry...${NC}"
gcloud auth configure-docker
echo -e "${GREEN}✅ Docker configured${NC}"

# Step 5: Build Docker image
echo -e "\n${YELLOW}Step 5: Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:latest .
echo -e "${GREEN}✅ Docker image built${NC}"

# Step 6: Push to GCR
echo -e "\n${YELLOW}Step 6: Pushing image to Google Container Registry...${NC}"
docker push ${IMAGE_NAME}:latest
echo -e "${GREEN}✅ Image pushed to GCR${NC}"

# Step 7: Deploy to Cloud Run
echo -e "\n${YELLOW}Step 7: Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory ${MEMORY} \
  --cpu ${CPU} \
  --timeout ${TIMEOUT} \
  --min-instances ${MIN_INSTANCES} \
  --max-instances ${MAX_INSTANCES} \
  --set-env-vars "ENVIRONMENT=production,CACHE_ENABLED=true,LOG_LEVEL=INFO,REDIS_HOST=localhost,REDIS_PORT=6379"

echo -e "${GREEN}✅ Deployed to Cloud Run${NC}"

# Step 8: Get service URL
echo -e "\n${YELLOW}Step 8: Retrieving service URL...${NC}"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --format 'value(status.url)')

echo -e "${GREEN}✅ Service URL: ${SERVICE_URL}${NC}"

# Step 9: Verify deployment
echo -e "\n${YELLOW}Step 9: Verifying deployment...${NC}"
sleep 5  # Wait for service to be ready

HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" ${SERVICE_URL}/health)
if [ "$HEALTH_CHECK" = "200" ]; then
    echo -e "${GREEN}✅ Health check passed (HTTP 200)${NC}"
else
    echo -e "${RED}❌ Health check failed (HTTP ${HEALTH_CHECK})${NC}"
    exit 1
fi

# Step 10: Display summary
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "Service URL: ${SERVICE_URL}"
echo -e "API Docs: ${SERVICE_URL}/docs"
echo -e "Health Check: ${SERVICE_URL}/health"
echo -e ""
echo -e "Example request:"
echo -e "curl -X POST ${SERVICE_URL}/recommend \\"
echo -e "  -H 'Content-Type: application/json' \\"
echo -e "  -d '{\"user_id\": 100, \"k\": 10, \"model\": \"mf\"}'"
echo -e "${GREEN}================================================${NC}"

# Step 11: Cost estimation
echo -e "\n${YELLOW}💰 Cost Estimation:${NC}"
echo -e "- Free tier: 2M requests/month"
echo -e "- After free tier: ~\$0.00002 per request"
echo -e "- Estimated cost for 1M requests/month: ~\$6/month"
echo -e "- Min instances = 0 means no cost when idle"
