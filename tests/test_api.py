"""Unit tests for FastAPI endpoints."""

import io
import cv2
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.data.generate_dataset import generate_base_metal_surface, inject_defect


client = TestClient(app)


def test_health_endpoint():
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "memory_usage_mb" in data
    assert "device" in data


def test_model_info_endpoint():
    """Test /model-info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "optimal_threshold" in data
    assert data["input_resolution"] == [128, 128, 3]


def test_predict_endpoint_with_image():
    """Test /predict with valid image."""
    img = generate_base_metal_surface(size=128)
    _, encoded = cv2.imencode(".png", img)
    
    files = {"file": ("test.png", io.BytesIO(encoded.tobytes()), "image/png")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "is_defective" in data
    assert "anomaly_score" in data
    assert "heatmap_base64" in data
    assert "latency_ms" in data


def test_predict_overlay_endpoint():
    """Test /predict/overlay returning image bytes."""
    img = generate_base_metal_surface(size=128)
    _, encoded = cv2.imencode(".png", img)
    
    files = {"file": ("test.png", io.BytesIO(encoded.tobytes()), "image/png")}
    response = client.post("/predict/overlay", files=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_sync_data_endpoint():
    """Test /sync-data endpoint."""
    response = client.post("/sync-data")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data
