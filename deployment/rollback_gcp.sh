#!/bin/bash

###############################################################################
# GCP Cloud Run Rollback Script
#
# Rolls back to a previous revision of the Cloud Run service
###############################################################################

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="clothing-recsys-api"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}Cloud Run Rollback${NC}"
echo -e "${YELLOW}================================================${NC}"

# Set project
gcloud config set project ${PROJECT_ID}

# List recent revisions
echo -e "\n${YELLOW}Recent revisions:${NC}"
gcloud run revisions list \
  --service ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --limit 5

# Get current revision
CURRENT_REVISION=$(gcloud run services describe ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --format 'value(status.latestReadyRevisionName)')

echo -e "\n${GREEN}Current revision: ${CURRENT_REVISION}${NC}"

# Get previous revision
PREVIOUS_REVISION=$(gcloud run revisions list \
  --service ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --limit 2 \
  --format 'value(metadata.name)' | tail -n 1)

echo -e "${YELLOW}Previous revision: ${PREVIOUS_REVISION}${NC}"

# Prompt for confirmation
read -p "Rollback to ${PREVIOUS_REVISION}? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Rollback cancelled${NC}"
    exit 1
fi

# Rollback
echo -e "\n${YELLOW}Rolling back...${NC}"
gcloud run services update-traffic ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --to-revisions ${PREVIOUS_REVISION}=100

echo -e "${GREEN}✅ Rollback complete${NC}"

# Verify
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --format 'value(status.url)')

sleep 5
HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" ${SERVICE_URL}/health)

if [ "$HEALTH_CHECK" = "200" ]; then
    echo -e "${GREEN}✅ Service healthy after rollback${NC}"
else
    echo -e "${RED}❌ Health check failed (HTTP ${HEALTH_CHECK})${NC}"
fi
