"""Sequential resilient downloader for BreastDCEDL I-SPY2 on Linux GCP VM.

Uses /usr/local/bin/ascli with unbuffered logging and auto-resume.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("/home/XiaoSha")
METADATA_FILE = BASE_DIR / "project" / "references" / "metadata" / "cases_all.csv"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
ASCLI_CMD = "/usr/local/bin/ascli"

FASPEX_URL = (
    "https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?"
    "context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLC"
    "JpZCI6IjExNTgiLCJwYXNzY29kZSI6IjVjZTYxMmQ3MmE2MTE1MDE0ZDRiZjdkNWRlYjI0OThiOWMzZDg1M2MiLC"
    "JwYWNrYWdlX2lkIjoiMTE1OCIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=="
)


def is_patient_complete(pid: str) -> bool:
    """Check if patient already has DCE and mask NIfTI files downloaded."""
    candidate_dirs = [
        RAW_DATA_DIR / pid,
        RAW_DATA_DIR / "PKG - BreastDCEDL_ISPY2" / "BreastDCEDL_ISPY2" / pid,
        RAW_DATA_DIR / "BreastDCEDL_ISPY2" / pid,
    ]
    for pdir in candidate_dirs:
        if pdir.is_dir():
            nii_files = list(pdir.glob("**/*.nii.gz"))
            if len(nii_files) >= 4:
                return True
    return False


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found: {METADATA_FILE}", flush=True)
        sys.exit(1)

    patients: list[str] = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"].strip()
            if pid and pid not in patients:
                patients.append(pid)

    total_patients = len(patients)
    print(f"=== Starting I-SPY2 Sequential Download on GCP VM ===", flush=True)
    print(f"Total I-SPY2 patients to process: {total_patients}", flush=True)
    print(f"Target directory: {RAW_DATA_DIR}", flush=True)

    downloaded = 0
    skipped = 0
    start_time = time.time()

    for i, pid in enumerate(patients, start=1):
        if is_patient_complete(pid):
            skipped += 1
            if skipped % 50 == 0:
                print(f"[{i}/{total_patients}] Skipped existing: {pid} (Total skipped: {skipped})", flush=True)
            continue

        target_remote_path = f"BreastDCEDL_ISPY2/{pid}"
        print(f"[{i}/{total_patients}] ({i*100/total_patients:.1f}%) Downloading {pid}...", flush=True)

        cmd = [
            ASCLI_CMD,
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
                    timeout=180,
                )
                if res.returncode == 0:
                    success = True
                    downloaded += 1
                    break
                else:
                    print(f"  Attempt {attempt} failed for {pid}: {res.stderr.strip()[:100]}", flush=True)
                    time.sleep(2 * attempt)
            except subprocess.TimeoutExpired:
                print(f"  Attempt {attempt} timed out for {pid}", flush=True)
                time.sleep(3)

        if not success:
            print(f"  [WARNING] Could not download {pid} after {retries} retries, moving to next.", flush=True)

        elapsed = time.time() - start_time
        processed = downloaded + skipped
        if downloaded > 0:
            avg_per_item = elapsed / downloaded
            remaining_items = total_patients - processed
            eta_mins = (avg_per_item * remaining_items) / 60
            print(f"  Done {pid} | Downloaded: {downloaded} | Skipped: {skipped} | ETA: {eta_mins:.1f} mins", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"I-SPY2 Sequential Download on GCP VM Completed!", flush=True)
    print(f"Total processed: {total_patients}", flush=True)
    print(f"Downloaded in this session: {downloaded}", flush=True)
    print(f"Previously existing: {skipped}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
