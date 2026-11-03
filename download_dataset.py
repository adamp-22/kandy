"""Download CIFAKE into this project's data directory."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("KAGGLEHUB_CACHE", str(PROJECT_ROOT / "data" / ".kagglehub"))

import kagglehub


if __name__ == "__main__":
    path = kagglehub.dataset_download(
        "birdy654/cifake-real-and-ai-generated-synthetic-images",
        output_dir=str(PROJECT_ROOT / "data" / "cifake"),
    )
    print("Path to dataset files:", path)
