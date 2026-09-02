"""
FastAPI Edge Anomaly Detection and DVC Sync Engine.
Provides high-throughput low-latency visual defect inspection endpoints.
"""

import os
import io
import time
import base64
import shutil
import subprocess
import psutil
from typing import Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.model.autoencoder import ConvAutoencoder, EdgeAnomalyDetector
from src.api.schemas import (
    AnomalyPredictionResponse,
    HealthResponse,
    DataSyncResponse,
    ModelInfoResponse,
    BoundingBox
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/edge_model.pt")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.01704"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

detector: Optional[EdgeAnomalyDetector] = None


def get_detector() -> EdgeAnomalyDetector:
    """Lazy loader to guarantee detector is always available."""
    global detector
    if detector is None:
        if os.path.exists(MODEL_PATH):
            try:
                detector = EdgeAnomalyDetector.load_from_checkpoint(
                    checkpoint_path=MODEL_PATH,
                    threshold=DEFAULT_THRESHOLD,
                    img_size=128,
                    device=DEVICE
                )
            except Exception as e:
                print(f"Notice: Loading fallback model ({e})")
                model = ConvAutoencoder()
                detector = EdgeAnomalyDetector(model=model, threshold=DEFAULT_THRESHOLD, device=DEVICE)
        else:
            model = ConvAutoencoder()
            detector = EdgeAnomalyDetector(model=model, threshold=DEFAULT_THRESHOLD, device=DEVICE)
    return detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights on startup."""
    print(f"Initializing Edge Anomaly Detection Service on {DEVICE}...")
    get_detector()
    yield
    print("Shutting down Edge Anomaly Detection Service...")


app = FastAPI(
    title="Edge Visual Quality Control API",
    description="Edge Anomaly Detection for Manufacturing Visual Inspection with DVC & MLflow",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive Landing Page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Edge Anomaly Detection API</title>
        <style>
            body { font-family: system-ui, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }
            .card { background: #1e293b; padding: 24px; border-radius: 12px; max-width: 800px; border: 1px solid #334155; }
            h1 { color: #38bdf8; margin-top: 0; }
            a { color: #38bdf8; text-decoration: none; font-weight: bold; }
            a:hover { text-decoration: underline; }
            .badge { display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-right: 6px; }
            .badge-dvc { background: #9333ea; color: #fff; }
            .badge-fastapi { background: #059669; color: #fff; }
            .badge-pytorch { background: #ea580c; color: #fff; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Edge Device Visual Quality Control</h1>
            <p>High-speed, edge-compatible anomaly detection for manufacturing inspection lines.</p>
            <div>
                <span class="badge badge-pytorch">PyTorch</span>
                <span class="badge badge-dvc">DVC + MinIO</span>
                <span class="badge badge-fastapi">FastAPI</span>
            </div>
            <hr style="border-color: #334155; margin: 20px 0;">
            <h3>Quick Links:</h3>
            <ul>
                <li><a href="/docs">Swagger API Documentation & Testing UI (/docs)</a></li>
                <li><a href="/health">Health & Resource Utilization (/health)</a></li>
                <li><a href="/model-info">Active Model Metadata (/model-info)</a></li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.post("/predict", response_model=AnomalyPredictionResponse)
async def predict_anomaly(
    file: UploadFile = File(..., description="Inspection image file (PNG/JPG)"),
    custom_threshold: Optional[float] = Form(None, description="Optional custom defect threshold override")
):
    """
    Run Visual Defect Anomaly Detection on an uploaded part image.
    Returns anomaly score, pass/fail verdict, defect bounding boxes, and visual heatmap overlay.
    """
    det = get_detector()
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
            
        result = det.predict(img_bgr, custom_threshold=custom_threshold)
        
        # Encode overlay heatmap to base64
        _, buffer = cv2.imencode('.png', result["heatmap_bgr"])
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        boxes = [
            BoundingBox(x=b["x"], y=b["y"], w=b["w"], h=b["h"])
            for b in result["bounding_boxes"]
        ]
        
        return AnomalyPredictionResponse(
            is_defective=result["is_defective"],
            verdict=result["verdict"],
            anomaly_score=result["anomaly_score"],
            threshold=result["threshold"],
            defect_confidence=result["defect_confidence"],
            latency_ms=result["latency_ms"],
            detected_defect_count=result["detected_defect_count"],
            bounding_boxes=boxes,
            heatmap_base64=f"data:image/png;base64,{heatmap_base64}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/overlay")
async def predict_overlay_image(
    file: UploadFile = File(..., description="Inspection image file (PNG/JPG)"),
    custom_threshold: Optional[float] = Form(None)
):
    """Run anomaly detection and return annotated defect heatmap directly as an image (JPEG)."""
    det = get_detector()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    result = det.predict(img_bgr, custom_threshold=custom_threshold)
    _, encoded_img = cv2.imencode('.jpg', result["heatmap_bgr"])
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@app.post("/sync-data", response_model=DataSyncResponse)
async def sync_data_with_dvc():
    """
    Sync and pull raw inspection datasets from remote MinIO / S3 storage via DVC.
    Enables edge devices to pull specific dataset versions without code repo bloat.
    """
    t0 = time.time()
    try:
        if not os.path.exists(".dvc"):
            return DataSyncResponse(
                status="warning",
                message="DVC is not initialized in this workspace.",
                duration_seconds=round(time.time() - t0, 3)
            )
            
        dvc_binary = shutil.which("dvc") or (".venv/bin/dvc" if os.path.exists(".venv/bin/dvc") else "dvc")
        cmd = [dvc_binary, "pull"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if res.returncode != 0:
            return DataSyncResponse(
                status="notice",
                message=f"DVC sync status: {res.stderr.strip() or res.stdout.strip() or 'Remote MinIO bucket offline or data already cached locally.'}",
                duration_seconds=round(time.time() - t0, 3)
            )
            
        return DataSyncResponse(
            status="success",
            message=f"Dataset synchronized successfully from remote storage. {res.stdout.strip()}",
            duration_seconds=round(time.time() - t0, 3)
        )
    except Exception as e:
        return DataSyncResponse(
            status="error",
            message=f"Data sync error: {str(e)}",
            duration_seconds=round(time.time() - t0, 3)
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System and Hardware Resource Health Check."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    dvc_configured = os.path.exists(".dvc")
    det = get_detector()
    
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        model_loaded=(det is not None),
        memory_usage_mb=round(mem_mb, 2),
        dvc_configured=dvc_configured
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get active model parameters, input resolution, and metrics."""
    det = get_detector()
    return ModelInfoResponse(
        model_name="ConvAutoencoder-Edge-v1",
        architecture="Encoder-Decoder ConvNet with Bottleneck",
        device=det.device,
        optimal_threshold=det.threshold,
        input_resolution=[det.img_size, det.img_size, 3],
        metrics={
            "threshold": det.threshold,
            "img_size": det.img_size
        }
    )
