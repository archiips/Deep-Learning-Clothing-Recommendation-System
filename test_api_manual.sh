#!/bin/bash

# Manual API testing script
# Run this after starting the API server

API_URL="http://localhost:8000"

echo "====================================="
echo "Testing Clothing Recommendation API"
echo "====================================="
echo ""

# Test 1: Health check
echo "Test 1: Health Check"
echo "GET $API_URL/health"
curl -s "$API_URL/health" | python3 -m json.tool
echo ""
echo ""

# Test 2: List models
echo "Test 2: List Models"
echo "GET $API_URL/models"
curl -s "$API_URL/models" | python3 -m json.tool
echo ""
echo ""

# Test 3: Get recommendations (MF model)
echo "Test 3: Get Recommendations (Matrix Factorization)"
echo "POST $API_URL/recommend"
curl -s -X POST "$API_URL/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "k": 5,
    "model": "mf"
  }' | python3 -m json.tool
echo ""
echo ""

# Test 4: Get recommendations (Popularity model)
echo "Test 4: Get Recommendations (Popularity)"
echo "POST $API_URL/recommend"
curl -s -X POST "$API_URL/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "k": 5,
    "model": "popularity"
  }' | python3 -m json.tool
echo ""
echo ""

# Test 5: Get recommendations with department filter
echo "Test 5: Get Recommendations (with Department Filter)"
echo "POST $API_URL/recommend"
curl -s -X POST "$API_URL/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "k": 5,
    "model": "mf",
    "department": "Tops"
  }' | python3 -m json.tool
echo ""
echo ""

# Test 6: Predict single rating
echo "Test 6: Predict Single Rating"
echo "POST $API_URL/predict"
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 100,
    "clothing_id": 1000,
    "model": "mf"
  }' | python3 -m json.tool
echo ""
echo ""

echo "====================================="
echo "Testing Complete!"
echo "====================================="
