"""Sequential resilient downloader for BreastDCEDL I-SPY2 dataset via Aspera CLI.

Reads references/metadata/cases_all.csv, checks existing downloads, and pulls
each patient individually to prevent Aspera buffer drain timeouts.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("D:/breastcancer_segmentation")
METADATA_FILE = BASE_DIR / "references" / "metadata" / "cases_all.csv"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
ASCLI_CMD = BASE_DIR / "tools" / "ascli" / "ascli.cmd"

FASPEX_URL = (
    "https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?"
    "context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLC"
    "JpZCI6IjExNTgiLCJwYXNzY29kZSI6IjVjZTYxMmQ3MmE2MTE1MDE0ZDRiZjdkNWRlYjI0OThiOWMzZDg1M2MiLC"
    "JwYWNrYWdlX2lkIjoiMTE1OCIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=="
)


def is_patient_complete(pid: str) -> bool:
    """Check if patient already has DCE and mask NIfTI files downloaded."""
    # Possible paths due to ascli folder nesting
    candidate_dirs = [
        RAW_DATA_DIR / pid,
        RAW_DATA_DIR / "PKG - BreastDCEDL_ISPY2" / "BreastDCEDL_ISPY2" / pid,
        RAW_DATA_DIR / "BreastDCEDL_ISPY2" / pid,
    ]
    for pdir in candidate_dirs:
        if pdir.is_dir():
            nii_files = list(pdir.glob("**/*.nii.gz"))
            # Each patient typically has 3-8 DCE files + 1 mask file
            if len(nii_files) >= 4:
                return True
    return False


def main() -> None:
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found: {METADATA_FILE}")
        sys.exit(1)

    patients: list[str] = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"].strip()
            if pid and pid not in patients:
                patients.append(pid)

    total_patients = len(patients)
    print(f"Total I-SPY2 patients to process: {total_patients}")

    downloaded = 0
    skipped = 0
    start_time = time.time()

    for i, pid in enumerate(patients, start=1):
        if is_patient_complete(pid):
            skipped += 1
            if skipped % 50 == 0 or skipped == total_patients:
                print(f"[{i}/{total_patients}] Skipped existing: {pid} (Total skipped: {skipped})")
            continue

        target_remote_path = f"BreastDCEDL_ISPY2/{pid}"
        print(f"[{i}/{total_patients}] ({i*100/total_patients:.1f}%) Downloading {pid}...")

        cmd = [
            str(ASCLI_CMD),
            "faspex5",
            "packages",
            "receive",
            f"--url={FASPEX_URL}",
            f"--to-folder={RAW_DATA_DIR}",
            target_remote_path,
        ]

        retries = 3
        success = False
        for attempt in range(1, retries + 1):
            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=120,
                )
                if res.returncode == 0:
                    success = True
                    downloaded += 1
                    break
                else:
                    print(f"  Attempt {attempt} failed for {pid}: {res.stderr.strip()[:100]}")
                    time.sleep(2 * attempt)
            except subprocess.TimeoutExpired:
                print(f"  Attempt {attempt} timed out for {pid}")
                time.sleep(3)

        if not success:
            print(f"  [WARNING] Could not download {pid} after {retries} retries, moving to next.")

        elapsed = time.time() - start_time
        processed = downloaded + skipped
        if downloaded > 0:
            avg_per_item = elapsed / downloaded
            remaining_items = total_patients - processed
            eta_mins = (avg_per_item * remaining_items) / 60
            print(f"  Done {pid} | Total downloaded: {downloaded} | ETA: {eta_mins:.1f} mins")

    print("\n" + "=" * 60)
    print(f"Sequential download completed!")
    print(f"Total processed: {total_patients}")
    print(f"Downloaded in this run: {downloaded}")
    print(f"Previously existing: {skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
