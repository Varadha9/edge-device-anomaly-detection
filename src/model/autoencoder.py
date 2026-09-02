"""
Lightweight Convolutional Autoencoder and Anomaly Detector for Edge Inspection.
Designed for high efficiency and low latency on edge computing devices.
"""

import time
from typing import Tuple, Dict, Any, List
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """
    Lightweight ConvAutoencoder for edge visual anomaly detection.
    Reconstructs normal manufacturing parts. Anomalous patterns (scratches, cracks)
    produce high reconstruction errors, revealing defective regions.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        
        # Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),          # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),         # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),        # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Bottleneck projection
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)
        
        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Sigmoid(),  # Normalizes output pixel values to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        feat = self.encoder(x)
        flat = self.flatten(feat)
        latent = self.fc_enc(flat)
        dec_in = self.fc_dec(latent).view(b, 256, 8, 8)
        reconstruction = self.decoder(dec_in)
        return reconstruction


class EdgeAnomalyDetector:
    """
    Inference Engine for Edge Device Anomaly Detection.
    Handles OpenCV preprocessing, reconstruction error calculation,
    anomaly score normalization, and visual heatmap generation.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.025,
        img_size: int = 128,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.img_size = img_size
        self.device = device

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        threshold: float = 0.025,
        img_size: int = 128,
        device: str = "cpu"
    ) -> "EdgeAnomalyDetector":
        """Load trained weights from checkpoint."""
        model = ConvAutoencoder()
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Support either full state dict or nested dict
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
            threshold = state_dict.get("threshold", threshold)
        else:
            model.load_state_dict(state_dict)
        return cls(model=model, threshold=threshold, img_size=img_size, device=device)

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess OpenCV BGR image to PyTorch tensor [1, 3, H, W] in range [0, 1]."""
        resized = cv2.resize(img_bgr, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor, resized

    def predict(
        self,
        img_bgr: np.ndarray,
        custom_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Run anomaly inspection on a single part image.
        Returns defect classification, anomaly score, and defect heatmap overlay.
        """
        t0 = time.perf_counter()
        th = custom_threshold if custom_threshold is not None else self.threshold
        safe_th = max(th, 1e-6)
        
        tensor, original_resized = self.preprocess(img_bgr)
        
        with torch.no_grad():
            reconstructed = self.model(tensor)
            
        # Pixel-wise Mean Squared Error
        error_map = torch.mean((tensor - reconstructed) ** 2, dim=1).squeeze(0).cpu().numpy()
        
        # Overall image anomaly score (mean of top 5% highest error pixels to capture localized defects)
        flat_errors = np.sort(error_map.flatten())
        top_k = max(1, int(len(flat_errors) * 0.05))
        anomaly_score = float(np.mean(flat_errors[-top_k:]))
        
        is_defective = bool(anomaly_score > th)
        
        # Generate defect visualization heatmap
        heatmap_overlay, bounding_boxes = self._generate_heatmap_overlay(
            original_resized, error_map, safe_th
        )
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        return {
            "is_defective": is_defective,
            "verdict": "DEFECTIVE" if is_defective else "PASS (NORMAL)",
            "anomaly_score": round(anomaly_score, 6),
            "threshold": round(th, 6),
            "defect_confidence": min(1.0, anomaly_score / (safe_th * 2.0)),
            "latency_ms": round(latency_ms, 2),
            "detected_defect_count": len(bounding_boxes),
            "bounding_boxes": bounding_boxes,
            "heatmap_bgr": heatmap_overlay,
        }

    def _generate_heatmap_overlay(
        self,
        original_bgr: np.ndarray,
        error_map: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """Create visual defect heatmap and identify defect contour bounding boxes."""
        safe_th = max(threshold, 1e-6)
        # Normalize error map for visual representation [0, 255]
        norm_error = np.clip((error_map / (safe_th * 2.5)) * 255.0, 0, 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(norm_error, cv2.COLORMAP_JET)
        
        # Alpha blend overlay
        overlay = cv2.addWeighted(original_bgr, 0.65, color_heatmap, 0.35, 0)
        
        # Find contours of regions exceeding defect threshold
        _, mask = cv2.threshold(norm_error, int(255 * 0.5), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:  # Filter out tiny noise dots
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                # Draw red defect bounding box on overlay
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    overlay, "DEFECT", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1
                )
                
        return overlay, bounding_boxes
