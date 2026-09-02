# Practical 11: Edge Device Anomaly Detection (Visual Quality Control)

An industrial-grade, edge-compatible Computer Vision Anomaly Detection and Dataset Management system for manufacturing inspection lines.

---

## 🏭 1. Problem Statement & Scenario

In automated manufacturing lines (e.g., metal casting, PCB assembly, automotive surface stamping), inspection cameras capture thousands of high-resolution images daily. However:
1. **Edge Storage is Constrained**: Edge devices (e.g., NVIDIA Jetson, Raspberry Pi, industrial IPCs) cannot store gigabytes of historical image archives locally.
2. **Binary Bloat in Git**: Committing image datasets directly into Git causes severe repository bloat and slows down deployments.
3. **Model & Data Lifecycle Synchronization**: Teams need to track which dataset version produced which model, evaluate anomaly detection metrics (ROC-AUC, reconstruction error), and deploy lightweight edge inference containers.

### Solution Architecture
- **DVC (Data Version Control) + MinIO / S3**: Version controls heavy image datasets remotely in an S3-compatible bucket while keeping lightweight `.dvc` hash pointer files in Git.
- **MLflow**: Tracks training parameters, reconstruction loss, ROC-AUC metrics, optimal anomaly threshold, and registers trained models.
- **PyTorch ConvAutoencoder**: Unsupervised visual anomaly detection model trained strictly on defect-free parts; anomalous structures (scratches, cracks, voids) produce high reconstruction error.
- **OpenCV**: Preprocessing, spatial error computation, defect bounding-box localization, and visual heatmap generation.
- **FastAPI**: Low-latency edge REST API serving `/predict`, `/predict/overlay`, `/sync-data`, `/health`, and `/model-info`.
- **Docker & Docker Compose**: Automated container packaging for edge nodes.

---

## 🏛️ 2. System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           CENTRAL STORAGE & MLOPS                       │
│                                                                         │
│  ┌───────────────────────┐                 ┌─────────────────────────┐  │
│  │     MinIO / S3        │                 │      MLflow Server      │  │
│  │  (Image Binaries)     │                 │   (Metrics & Models)    │  │
│  └──────────▲────────────┘                 └────────────▲────────────┘  │
└─────────────┼───────────────────────────────────────────┼───────────────┘
              │ dvc push / dvc pull                       │ Log Metrics
              ▼                                           │
┌─────────────────────────────────────────────────────────┴───────────────┐
│                       EDGE INSPECTION CONTAINER (Docker)                │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                       FastAPI Engine (Port 8000)                  │  │
│  │                                                                   │  │
│  │   POST /predict  ──►  OpenCV Preprocessing                        │  │
│  │                              │                                    │  │
│  │                              ▼                                    │  │
│  │                       PyTorch ConvAutoencoder                     │  │
│  │                              │                                    │  │
│  │                              ▼                                    │  │
│  │                       Anomaly Error & Heatmap Engine              │  │
│  │                              │                                    │  │
│  │   POST /sync-data ──► DVC Client Sync from MinIO                  │  │
│  │   GET  /health    ──► CPU / RAM / Device Monitoring               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 3. Directory Structure

```text
edge-anomaly-detection/
├── data/
│   ├── raw/
│   │   ├── train/normal/      # Defect-free parts for training
│   │   ├── test/normal/       # Normal test samples
│   │   └── test/defective/    # Defective test samples (scratches, cracks, voids, stains)
│   └── demo_output/           # Inspection output heatmaps
├── models/
│   ├── edge_model.pt          # PyTorch trained model checkpoint
│   └── model_metadata.json    # MLflow run info & optimal threshold
├── scripts/
│   ├── create_minio_bucket.py # MinIO S3 bucket initializer
│   └── setup_dvc_minio.sh     # DVC remote configuration script
├── src/
│   ├── api/
│   │   ├── app.py             # FastAPI edge inference & sync server
│   │   └── schemas.py         # Pydantic request/response models
│   ├── data/
│   │   └── generate_dataset.py # Synthetic manufacturing image generator
│   └── model/
│       └── autoencoder.py     # ConvAutoencoder & EdgeAnomalyDetector
├── tests/
│   ├── test_api.py            # API endpoint unit tests
│   └── test_model.py          # Model architecture & detector tests
├── demo.py                    # End-to-end automated demo script
├── Dockerfile                 # Multi-stage lightweight edge Dockerfile
├── docker-compose.yml         # Full stack: MinIO + MLflow + Edge API
├── Makefile                   # Quick execution targets
├── requirements.txt           # Python dependencies
└── README.md                  # Practical documentation
```

---

## 🚀 4. Quickstart Guide

### Step 1: Environment Setup
```bash
# Create virtual environment and install dependencies
make setup
# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Generate Manufacturing Dataset
```bash
# Generates train/test split with normal parts and synthetic defect patterns
python src/data/generate_dataset.py --train-normal 120 --test-normal 30 --test-defective 30
```

### Step 3: Configure DVC & MinIO Remote
```bash
# Initialize DVC and point remote to MinIO S3 bucket
bash scripts/setup_dvc_minio.sh
```

### Step 4: Train ConvAutoencoder & Track with MLflow
```bash
python src/train.py --epochs 20 --batch-size 16 --lr 0.001
```

### Step 5: Run Automated End-to-End Demo
```bash
python demo.py
```

### Step 6: Start Edge FastAPI Service
```bash
# Start local development server on port 8000
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🐳 5. Running with Docker Compose

Launch the complete containerized stack (MinIO Object Storage + MLflow Server + Edge Inspection API):

```bash
# Build and run containers in background
docker compose up -d

# Check status of containers
docker compose ps
```

Services will be accessible at:
- **FastAPI Edge Service**: [http://localhost:8000](http://localhost:8000) (Interactive Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs))
- **MinIO S3 Console**: [http://localhost:9001](http://localhost:9001) (User: `minioadmin` / Password: `minioadmin`)
- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)

---

## 📡 6. API Reference

### 1. `POST /predict`
Uploads a part image for defect inspection.
- **Request**: Multipart Form Data (`file`: Image PNG/JPEG)
- **Response**:
```json
{
  "is_defective": true,
  "verdict": "DEFECTIVE",
  "anomaly_score": 0.048123,
  "threshold": 0.024510,
  "defect_confidence": 0.98,
  "latency_ms": 12.4,
  "detected_defect_count": 2,
  "bounding_boxes": [
    {"x": 42, "y": 58, "w": 34, "h": 18}
  ],
  "heatmap_base64": "data:image/png;base64,iVBORw0KGgo..."
}
```

### 2. `POST /predict/overlay`
Returns the annotated image directly as JPEG image stream with bounding boxes and defect heatmap overlaid.

### 3. `POST /sync-data`
Triggers DVC pull to sync latest dataset binaries from MinIO remote without code repository updates.

### 4. `GET /health`
Returns hardware resource utilization (RAM MB, CPU, loaded model status).

### 5. `GET /model-info`
Returns current model parameters, input resolution, and operational anomaly threshold.

---

## 🧪 7. Running Tests

```bash
pytest tests/ -v
```

---

## 🎓 8. Practical Exam & Viva Questions Guide

### Q1: Why use DVC instead of committing images to Git?
> **Answer**: Git is optimized for small text files and diffs. Storing large binary datasets (images, video, weights) causes the `.git` directory to bloat permanently, making `git clone` and `git pull` extremely slow. DVC stores image binaries in remote object storage (like MinIO/S3) and only commits small text pointer files (`.dvc` containing md5 hashes) to Git.

### Q2: Why is an Autoencoder used for manufacturing visual inspection instead of a standard classifier?
> **Answer**: In manufacturing lines, defect samples are rare, varied, and unpredictable (new defect types can appear anytime). Training a standard supervised binary classifier requires balanced defect data. An Autoencoder is trained **only on normal defect-free parts** (unsupervised anomaly detection). When presented with a defective part, the model cannot reconstruct the unseen defect pattern, resulting in a high reconstruction error (MSE) that accurately flags and localizes the anomaly.

### Q3: How is the anomaly score and spatial localization calculated?
> **Answer**:
> 1. Input image $X$ is passed through the ConvAutoencoder to obtain reconstruction $\hat{X}$.
> 2. The pixel-wise squared error map $E = (X - \hat{X})^2$ is computed.
> 3. The anomaly score is computed using the mean of the top 5% error pixels to capture localized scratches or cracks without being diluted by the large background.
> 4. OpenCV applies a COLORMAP_JET color overlay and contour extraction to draw bounding boxes around defective zones.

### Q4: Why is MinIO used alongside DVC?
> **Answer**: MinIO is an open-source, high-performance, S3-compatible object storage server. It allows local on-premise simulation and private cloud deployments of S3 storage, enabling edge devices to sync data seamlessly via standard S3 APIs.
