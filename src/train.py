"""
Model Training and Experiment Tracking with MLflow.
Trains lightweight ConvAutoencoder on normal manufacturing parts and evaluates
ROC-AUC on test normal vs defective sets.
"""

import os
import sys
import glob
import json
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import mlflow
import mlflow.pytorch

from src.model.autoencoder import ConvAutoencoder, EdgeAnomalyDetector


class ManufacturingDataset(Dataset):
    """Custom Dataset loading images from directory."""

    def __init__(self, image_paths, img_size=128, transform=None):
        self.image_paths = image_paths
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {path}")
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (self.img_size, self.img_size)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        if self.transform:
            tensor = self.transform(tensor)
        return tensor


def train_anomaly_model(
    data_dir: str = "data/raw",
    output_model_path: str = "models/edge_model.pt",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 64,
    img_size: int = 128,
    mlflow_tracking_uri: str = None,
    experiment_name: str = "edge-anomaly-detection"
):
    """Train ConvAutoencoder and track with MLflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configure MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    else:
        # Default to local sqlite database
        db_path = Path("mlflow.db").resolve()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
    mlflow.set_experiment(experiment_name)
    
    # Locate dataset
    train_normal_files = glob.glob(f"{data_dir}/train/normal/*.png")
    test_normal_files = glob.glob(f"{data_dir}/test/normal/*.png")
    test_defective_files = glob.glob(f"{data_dir}/test/defective/*.png")
    
    if not train_normal_files:
        raise FileNotFoundError(
            f"No training images found in {data_dir}/train/normal/. Run dataset generator first!"
        )
        
    print(f"Found {len(train_normal_files)} train normal images, "
          f"{len(test_normal_files)} test normal, {len(test_defective_files)} test defective.")
          
    # Augmentations for normal parts
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    train_dataset = ManufacturingDataset(train_normal_files, img_size=img_size, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize Model & Optimizer
    model = ConvAutoencoder(in_channels=3, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    with mlflow.start_run(run_name=f"run_ae_{int(time.time())}") as run:
        # Log Parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "latent_dim": latent_dim,
            "image_size": img_size,
            "device": str(device),
            "train_samples": len(train_normal_files),
        })
        
        print("Starting training loop...")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for batch_images in train_loader:
                batch_images = batch_images.to(device)
                optimizer.zero_grad()
                reconstructed = model(batch_images)
                loss = criterion(reconstructed, batch_images)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_images.size(0)
                
            epoch_loss = running_loss / len(train_dataset)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            
            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.6f}")
                
        training_duration = time.time() - start_time
        mlflow.log_metric("training_duration_sec", training_duration)
        print(f"Training finished in {training_duration:.2f} seconds.")
        
        # Evaluation & Anomaly Threshold Calibration
        print("Evaluating on test set and calculating optimal anomaly threshold...")
        model.eval()
        detector = EdgeAnomalyDetector(model=model, threshold=0.01, img_size=img_size, device=str(device))
        
        test_scores = []
        test_labels = []  # 0 for normal, 1 for defective
        
        for p in test_normal_files:
            img = cv2.imread(p)
            res = detector.predict(img)
            test_scores.append(res["anomaly_score"])
            test_labels.append(0)
            
        for p in test_defective_files:
            img = cv2.imread(p)
            res = detector.predict(img)
            test_scores.append(res["anomaly_score"])
            test_labels.append(1)
            
        test_scores = np.array(test_scores)
        test_labels = np.array(test_labels)
        
        roc_auc = float(roc_auc_score(test_labels, test_scores))
        
        # Calculate optimal threshold based on maximum F1-score
        candidate_thresholds = np.linspace(test_scores.min(), test_scores.max(), 100)
        best_f1, best_threshold, best_prec, best_rec = 0.0, float(np.median(test_scores)), 0.0, 0.0
        
        for th in candidate_thresholds:
            preds = (test_scores > th).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(test_labels, preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold, best_prec, best_rec = float(f1), float(th), float(prec), float(rec)
                
        print(f"✓ Evaluation Results:")
        print(f"  - ROC-AUC: {roc_auc:.4f}")
        print(f"  - Best F1-Score: {best_f1:.4f}")
        print(f"  - Optimal Threshold: {best_threshold:.6f}")
        print(f"  - Precision: {best_prec:.4f}, Recall: {best_rec:.4f}")
        
        # Log Metrics
        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "f1_score": best_f1,
            "precision": best_prec,
            "recall": best_rec,
            "optimal_threshold": best_threshold,
            "normal_score_mean": float(np.mean(test_scores[test_labels == 0])),
            "defective_score_mean": float(np.mean(test_scores[test_labels == 1])),
        })
        
        # Save PyTorch checkpoint
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "img_size": img_size,
            "threshold": best_threshold,
            "metrics": {
                "roc_auc": roc_auc,
                "f1_score": best_f1,
            }
        }, output_model_path)
        print(f"Saved model checkpoint to {output_model_path}")
        
        # Log model artifact to MLflow
        mlflow.log_artifact(output_model_path, artifact_path="model_weights")
        try:
            example_input = torch.randn(1, 3, img_size, img_size).to(device)
            mlflow.pytorch.log_model(model, name="edge_autoencoder_model", input_example=example_input.cpu().numpy())
        except Exception as e:
            print(f"MLflow model flavor notice: {e}")
            
        metadata = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "optimal_threshold": best_threshold,
            "roc_auc": roc_auc,
            "f1_score": best_f1,
            "saved_model_path": output_model_path,
        }
        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="models/edge_model.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--mlflow-uri", default=None)
    args = parser.parse_args()
    
    train_anomaly_model(
        data_dir=args.data_dir,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        mlflow_tracking_uri=args.mlflow_uri
    )
