"""Pydantic schemas for FastAPI edge anomaly detection service."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    w: int = Field(..., description="Bounding box width")
    h: int = Field(..., description="Bounding box height")


class AnomalyPredictionResponse(BaseModel):
    is_defective: bool = Field(..., description="True if anomaly score exceeds threshold")
    verdict: str = Field(..., description="Inspection verdict: 'PASS (NORMAL)' or 'DEFECTIVE'")
    anomaly_score: float = Field(..., description="Calculated visual reconstruction error score")
    threshold: float = Field(..., description="Defect decision boundary threshold")
    defect_confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    latency_ms: float = Field(..., description="Inference execution time in milliseconds")
    detected_defect_count: int = Field(..., description="Count of detected anomalous regions")
    bounding_boxes: List[BoundingBox] = Field(default=[], description="Defect bounding boxes")
    heatmap_base64: Optional[str] = Field(None, description="Base64 PNG of defect heatmap overlay")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="API server status")
    device: str = Field(..., description="Execution device (CPU/CUDA)")
    model_loaded: bool = Field(..., description="Whether anomaly detection model is loaded")
    memory_usage_mb: float = Field(..., description="Memory utilization in MB")
    dvc_configured: bool = Field(..., description="Whether DVC remote is configured")


class DataSyncResponse(BaseModel):
    status: str
    message: str
    synced_files_count: Optional[int] = 0
    duration_seconds: Optional[float] = 0.0


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    device: str
    optimal_threshold: float
    input_resolution: List[int]
    metrics: Optional[Dict[str, Any]] = None
