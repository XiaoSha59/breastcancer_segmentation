"""Upload curated 5-channel I-SPY2 dataset to Kaggle as Private Dataset."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path("/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2")
KAGGLE_CMD = "/home/XiaoSha/.local/bin/kaggle"
CREDS_FILE = Path("/home/XiaoSha/.kaggle/kaggle.json")


def main() -> None:
    if not CREDS_FILE.exists():
        print(f"Error: Credentials not found at {CREDS_FILE}")
        sys.exit(1)

    with open(CREDS_FILE, "r", encoding="utf-8") as f:
        creds = json.load(f)
    username = creds.get("username")
    print(f"Authenticated as Kaggle user: {username}")

    slug = "ispy2-breast-dce-mri-5channel"
    meta_file = DATASET_DIR / "dataset-metadata.json"

    meta_content = {
        "title": "ISPY2 Breast DCE-MRI 5-Channel Curated",
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_content, f, indent=2)
    print(f"Created metadata at: {meta_file}")

    print("\nStarting Kaggle dataset upload (Private)...")
    print(f"Source folder: {DATASET_DIR}")
    print("Mode: uncompressed tar (fastest for .nii.gz)")

    cmd = [
        KAGGLE_CMD,
        "datasets",
        "create",
        "-p",
        str(DATASET_DIR),
        "-r",
        "tar",
    ]

    res = subprocess.run(cmd, text=True)
    if res.returncode == 0:
        print("\n=== Dataset successfully created on Kaggle! ===")
        print(f"URL: https://www.kaggle.com/datasets/{username}/{slug}")
    else:
        print(f"\nUpload process finished with code {res.returncode}")


if __name__ == "__main__":
    main()
