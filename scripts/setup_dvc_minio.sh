#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if command -v dvc &> /dev/null; then
    DVC_CMD="dvc"
elif [ -f ".venv/bin/dvc" ]; then
    DVC_CMD=".venv/bin/dvc"
else
    echo "Error: DVC not found in PATH or .venv/bin/dvc"
    exit 1
fi

echo "=== Setting up DVC with MinIO S3 Backend ==="

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    git config user.name "EdgeEngineer"
    git config user.email "engineer@edge-qc.local"
fi

# Initialize DVC
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    $DVC_CMD init -f --no-scm || $DVC_CMD init -f
fi

# Configure MinIO Remote in DVC
echo "Configuring MinIO remote in DVC..."
$DVC_CMD remote add -d -f minio_remote s3://edge-data-bucket
$DVC_CMD remote modify minio_remote endpointurl http://localhost:9000
$DVC_CMD remote modify minio_remote access_key_id minioadmin
$DVC_CMD remote modify minio_remote secret_access_key minioadmin
$DVC_CMD remote modify minio_remote use_ssl false

echo "✓ DVC remote configuration completed:"
$DVC_CMD remote list

# If raw data exists, track it with DVC
if [ -d "data/raw" ]; then
    echo "Tracking raw dataset with DVC..."
    $DVC_CMD add data/raw
    echo "✓ Tracked data/raw with DVC (data/raw.dvc created)"
    git add data/.gitignore data/raw.dvc .dvc/config 2>/dev/null || true
fi

echo "=== DVC Setup Finished ==="
