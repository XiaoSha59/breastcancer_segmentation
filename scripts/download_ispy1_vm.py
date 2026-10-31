"""Download and extract BreastDCEDL I-SPY1 (Full package ~2.8 GB) on GCP VM."""

import os
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path("/home/XiaoSha/data/raw")
ZIP_PATH = DATA_DIR / "BreastDCEDL_spy1.zip"
EXTRACT_DIR = DATA_DIR / "BreastDCEDL_spy1"

URL_ZIP = "https://zenodo.org/records/18114231/files/BreastDCEDL_spy1.zip?download=1"
URL_META = "https://zenodo.org/records/18114231/files/BreastDCEDL_metadata_min_crop.csv?download=1"
META_PATH = DATA_DIR / "BreastDCEDL_metadata_min_crop.csv"


def download_and_extract() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download metadata
    if not META_PATH.exists():
        print("Downloading metadata CSV...", flush=True)
        urllib.request.urlretrieve(URL_META, META_PATH)
        print("Metadata downloaded.", flush=True)

    if EXTRACT_DIR.is_dir() and len(list(EXTRACT_DIR.iterdir())) >= 100:
        print(f"I-SPY1 full dataset already extracted at {EXTRACT_DIR}", flush=True)
        return

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 2_500_000_000:
        print(f"Downloading I-SPY1 full package (2.79 GB) from {URL_ZIP}...", flush=True)

        def report(count, block_size, total_size):
            if count % 3000 == 0:
                downloaded = count * block_size / (1024 * 1024)
                total = total_size / (1024 * 1024)
                print(f"  {downloaded:.1f} MB / {total:.1f} MB ({downloaded*100/total:.1f}%)", flush=True)

        urllib.request.urlretrieve(URL_ZIP, ZIP_PATH, reporthook=report)
        print("Download complete!", flush=True)

    print(f"Extracting {ZIP_PATH} to {DATA_DIR}...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete!", flush=True)


if __name__ == "__main__":
    download_and_extract()
