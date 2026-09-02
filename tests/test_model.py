"""Unit tests for ConvAutoencoder and Edge Anomaly Detector."""

import numpy as np
import torch
import pytest

from src.model.autoencoder import ConvAutoencoder, EdgeAnomalyDetector
from src.data.generate_dataset import generate_base_metal_surface, inject_defect


def test_autoencoder_architecture():
    """Test autoencoder forward pass tensor dimensions."""
    model = ConvAutoencoder(in_channels=3, latent_dim=64)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 3, 128, 128), f"Expected (2, 3, 128, 128) but got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Output must be within [0, 1] range"


def test_anomaly_detector_prediction_normal():
    """Test anomaly detector on normal sample."""
    model = ConvAutoencoder(in_channels=3, latent_dim=64)
    detector = EdgeAnomalyDetector(model=model, threshold=0.05, img_size=128, device="cpu")
    
    normal_img = generate_base_metal_surface(size=128)
    result = detector.predict(normal_img)
    
    assert "is_defective" in result
    assert "anomaly_score" in result
    assert "verdict" in result
    assert "latency_ms" in result
    assert "heatmap_bgr" in result
    assert isinstance(result["anomaly_score"], float)
    assert result["heatmap_bgr"].shape == (128, 128, 3)


def test_anomaly_detector_prediction_defect():
    """Test anomaly detector on defective sample."""
    model = ConvAutoencoder(in_channels=3, latent_dim=64)
    detector = EdgeAnomalyDetector(model=model, threshold=0.0001, img_size=128, device="cpu")
    
    base_img = generate_base_metal_surface(size=128)
    defective_img, defect_type = inject_defect(base_img)
    
    result = detector.predict(defective_img)
    assert result["is_defective"] is True
    assert result["verdict"] == "DEFECTIVE"
    assert result["heatmap_bgr"].shape == (128, 128, 3)
