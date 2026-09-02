.PHONY: help setup data dvc-setup train test api docker-build up down demo

VENV = .venv/bin
PYTHON = $(VENV)/python
PIP = $(VENV)/pip
PYTEST = $(VENV)/pytest
UVICORN = $(VENV)/uvicorn

help:
	@echo "Edge Device Anomaly Detection (Visual Quality Control) Commands:"
	@echo "  make setup        - Create virtualenv and install all dependencies"
	@echo "  make data         - Generate visual quality control dataset"
	@echo "  make dvc-setup    - Configure DVC with MinIO S3 backend"
	@echo "  make train        - Train ConvAutoencoder and log metrics to MLflow"
	@echo "  make test         - Run test suite"
	@echo "  make api          - Run local FastAPI edge inference server"
	@echo "  make demo         - Run end-to-end pipeline demo"
	@echo "  make up           - Start Docker Compose stack (MinIO + MLflow + Edge API)"
	@echo "  make down         - Stop Docker Compose stack"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) src/data/generate_dataset.py

dvc-setup:
	bash scripts/setup_dvc_minio.sh

train:
	$(PYTHON) src/train.py

test:
	$(PYTEST) tests/ -v

api:
	$(UVICORN) src.api.app:app --host 0.0.0.0 --port 8000 --reload

demo:
	$(PYTHON) demo.py

docker-build:
	docker build -t edge-cv-api:latest .

up:
	docker compose up -d

down:
	docker compose down
