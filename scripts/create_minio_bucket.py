"""Helper script to initialize MinIO bucket for DVC storage."""
import os
import boto3
from botocore.client import Config

ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "edge-data-bucket")

def init_bucket():
    print(f"Connecting to MinIO at {ENDPOINT}...")
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )
    
    try:
        buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
        if BUCKET not in buckets:
            print(f"Creating bucket: {BUCKET}")
            s3.create_bucket(Bucket=BUCKET)
            print(f"✓ Bucket '{BUCKET}' created successfully.")
        else:
            print(f"✓ Bucket '{BUCKET}' already exists.")
    except Exception as e:
        print(f"MinIO connection/bucket notice: {e}")

if __name__ == "__main__":
    init_bucket()
