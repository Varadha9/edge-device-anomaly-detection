"""
End-to-End Demonstration Script for Edge Visual Quality Control.
Generates test data, trains model, tests normal vs defective samples,
and outputs visual inspection heatmaps.
"""

import os
import sys
import glob
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch

from src.data.generate_dataset import create_inspection_dataset
from src.train import train_anomaly_model
from src.model.autoencoder import EdgeAnomalyDetector


def run_demo():
    print("\n" + "="*65)
    print("  EDGE DEVICE VISUAL ANOMALY DETECTION (QUALITY CONTROL)")
    print("="*65)
    
    # 1. Dataset Generation
    data_dir = "data/raw"
    if not os.path.exists(f"{data_dir}/train/normal"):
        print("\n[Step 1/4] Generating Manufacturing Inspection Dataset...")
        create_inspection_dataset(output_dir=data_dir, num_train_normal=100, num_test_normal=25, num_test_defective=25)
    else:
        print("\n[Step 1/4] Dataset already exists in data/raw.")
        
    # 2. Train Model & Track with MLflow
    model_path = "models/edge_model.pt"
    print("\n[Step 2/4] Training Lightweight ConvAutoencoder with MLflow Tracking...")
    train_anomaly_model(
        data_dir=data_dir,
        output_model_path=model_path,
        epochs=15,
        batch_size=16,
        lr=1e-3,
        latent_dim=64
    )
    
    # 3. Load Trained Edge Detector
    print("\n[Step 3/4] Initializing Edge Inference Engine...")
    detector = EdgeAnomalyDetector.load_from_checkpoint(model_path, device="cpu")
    print(f"✓ Detector ready. Operational threshold: {detector.threshold:.6f}")
    
    # 4. Perform Quality Control Testing
    print("\n[Step 4/4] Executing Visual Quality Inspection on Sample Parts...")
    demo_out_dir = Path("data/demo_output")
    demo_out_dir.mkdir(parents=True, exist_ok=True)
    
    test_normal = sorted(glob.glob(f"{data_dir}/test/normal/*.png"))[:3]
    test_defective = sorted(glob.glob(f"{data_dir}/test/defective/*.png"))[:4]
    
    print("\n" + "-"*78)
    print(f"{'Sample Image':<32} | {'Score':<10} | {'Threshold':<10} | {'Verdict':<14} | {'Latency':<8}")
    print("-" * 78)
    
    # Evaluate Normal Samples
    for p in test_normal:
        img = cv2.imread(p)
        res = detector.predict(img)
        fname = Path(p).name
        print(f"{fname:<32} | {res['anomaly_score']:<10.6f} | {res['threshold']:<10.6f} | {res['verdict']:<14} | {res['latency_ms']:<6.1f}ms")
        cv2.imwrite(str(demo_out_dir / f"annotated_{fname}"), res["heatmap_bgr"])
        
    # Evaluate Defective Samples
    for p in test_defective:
        img = cv2.imread(p)
        res = detector.predict(img)
        fname = Path(p).name
        print(f"{fname:<32} | {res['anomaly_score']:<10.6f} | {res['threshold']:<10.6f} | {res['verdict']:<14} | {res['latency_ms']:<6.1f}ms")
        cv2.imwrite(str(demo_out_dir / f"annotated_{fname}"), res["heatmap_bgr"])
        
    print("-" * 78)
    print(f"\n✓ Inspection completed! Visual defect heatmaps saved to '{demo_out_dir}/'")
    print("="*65 + "\n")


if __name__ == "__main__":
    run_demo()
