"""
Unit tests for the recommendation API.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app

# Create test client
client = TestClient(app)


class TestGeneralEndpoints:
    """Test general API endpoints."""

    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "docs" in data
        assert "health" in data

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert data["num_users"] > 0
        assert data["num_items"] > 0
        assert len(data["available_models"]) == 3

    def test_list_models(self):
        """Test models listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 3
        assert data["default_model"] in ["popularity", "mf", "ncf"]

        # Check model names
        model_names = [m["model_name"] for m in data["models"]]
        assert "popularity" in model_names
        assert "mf" in model_names
        assert "ncf" in model_names


class TestRecommendations:
    """Test recommendation endpoints."""

    def test_recommend_default(self):
        """Test recommendation with default parameters."""
        response = client.post("/recommend", json={
            "user_id": 100,
            "k": 10,
            "model": "mf"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 100
        assert len(data["recommendations"]) <= 10
        assert data["model_used"] == "mf"
        assert "timestamp" in data

    def test_recommend_popularity(self):
        """Test popularity-based recommendations."""
        response = client.post("/recommend", json={
            "user_id": 100,
            "k": 5,
            "model": "popularity"
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) <= 5
        assert data["model_used"] == "popularity"

    def test_recommend_ncf(self):
        """Test NCF model recommendations."""
        response = client.post("/recommend", json={
            "user_id": 100,
            "k": 10,
            "model": "ncf"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == "ncf"
        assert len(data["recommendations"]) <= 10

    def test_recommend_with_department_filter(self):
        """Test recommendations with department filter."""
        response = client.post("/recommend", json={
            "user_id": 100,
            "k": 10,
            "model": "mf",
            "department": "Tops"
        })
        assert response.status_code == 200
        data = response.json()
        # Check that all recommendations are from Tops department
        for rec in data["recommendations"]:
            assert rec["department"] == "Tops"

    def test_recommend_invalid_model(self):
        """Test recommendation with invalid model."""
        response = client.post("/recommend", json={
            "user_id": 100,
            "k": 10,
            "model": "invalid_model"
        })
        assert response.status_code == 400

    def test_recommend_invalid_k(self):
        """Test recommendation with invalid k value."""
        response = client.post("/recommend", json={
            "user_id": 100,
            "k": 50,  # Exceeds max of 20
            "model": "mf"
        })
        assert response.status_code == 422  # Validation error

    def test_recommendation_structure(self):
        """Test recommendation item structure."""
        response = client.post("/recommend", json={
            "user_id": 100,
            "k": 5,
            "model": "mf"
        })
        assert response.status_code == 200
        data = response.json()

        # Check first recommendation has all required fields
        if len(data["recommendations"]) > 0:
            rec = data["recommendations"][0]
            assert "clothing_id" in rec
            assert "predicted_score" in rec
            assert "rank" in rec
            assert "department" in rec
            assert "class_name" in rec
            assert "avg_rating" in rec
            assert "num_reviews" in rec
            assert rec["rank"] == 1  # First item should have rank 1


class TestPredictions:
    """Test prediction endpoints."""

    def test_predict_rating(self):
        """Test single rating prediction."""
        response = client.post("/predict", json={
            "user_id": 100,
            "clothing_id": 1000,
            "model": "mf"
        })
        # May return 404 if clothing_id doesn't exist, which is valid
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert data["user_id"] == 100
            assert data["clothing_id"] == 1000
            assert "predicted_rating" in data
            assert data["model_used"] == "mf"
            assert "timestamp" in data

    def test_predict_invalid_user(self):
        """Test prediction with invalid user."""
        response = client.post("/predict", json={
            "user_id": 999999,  # Non-existent user
            "clothing_id": 1000,
            "model": "mf"
        })
        assert response.status_code == 404

    def test_predict_invalid_item(self):
        """Test prediction with invalid item."""
        response = client.post("/predict", json={
            "user_id": 100,
            "clothing_id": 999999,  # Non-existent item
            "model": "mf"
        })
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
