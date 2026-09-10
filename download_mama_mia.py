# =============================================================================
# Download MAMA-MIA MRI images (non-I-SPY2 external cohorts)
#
# Downloads Duke (291) + I-SPY1 (171) + NACT-Pilot (64) = 526 cases.
# MAMA-MIA I-SPY2-source cases are skipped because they overlap the development
# source cohort used in this study.
#
# SECURITY:
#   Do NOT place a Synapse token in this file.
#   Set it in the environment variable SYNAPSE_AUTH_TOKEN.
#
# Optional:
#   Set MAMA_MIA_IMAGES_DIR to choose the local download directory.
#   Otherwise images are written to ./images next to this script.
# =============================================================================

from __future__ import annotations

import os
import sys
from pathlib import Path

import synapseclient


SYN_IMAGES = "syn64871114"

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = Path(
    os.environ.get("MAMA_MIA_IMAGES_DIR", str(PROJECT_ROOT / "images"))
).expanduser()

SYNAPSE_TOKEN = os.environ.get("SYNAPSE_AUTH_TOKEN", "").strip()


def main() -> int:
    if not SYNAPSE_TOKEN:
        print(
            "ERROR: SYNAPSE_AUTH_TOKEN is not set.\n"
            "Create/use your own Synapse personal access token and expose it only "
            "through the environment variable SYNAPSE_AUTH_TOKEN."
        )
        return 2

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.login(authToken=SYNAPSE_TOKEN, silent=True)
    print("Synapse connection successful.")

    print("Scanning MAMA-MIA image folders...")
    all_cases = list(syn.getChildren(SYN_IMAGES))

    non_ispy2 = [
        c
        for c in all_cases
        if not str(c["name"]).strip().upper().startswith("ISPY2")
    ]

    print(f"Total folders in MAMA-MIA images: {len(all_cases)}")
    print(f"Non-I-SPY2 external cases to download: {len(non_ispy2)}")
    if len(non_ispy2) != 526:
        print(
            "WARNING: expected 526 non-I-SPY2 cases for the dataset version "
            "used in the manuscript."
        )

    done_cases = 0
    done_files = 0
    skipped_files = 0
    error_files = 0

    for i, case in enumerate(non_ispy2, start=1):
        case_name = str(case["name"]).strip()
        case_dir = IMAGES_DIR / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = list(syn.getChildren(case["id"]))
        except Exception as exc:
            print(f"[{i:3d}/{len(non_ispy2)}] ERROR reading {case_name}: {exc}")
            error_files += 1
            continue

        already = {p.name for p in case_dir.iterdir() if p.is_file()}
        needed = [fobj for fobj in files if fobj["name"] not in already]

        if not needed:
            print(
                f"[{i:3d}/{len(non_ispy2)}] already present: "
                f"{case_name} ({len(files)} files)"
            )
            skipped_files += len(files)
            done_cases += 1
            continue

        print(
            f"[{i:3d}/{len(non_ispy2)}] {case_name}: "
            f"{len(already)} present, downloading {len(needed)}"
        )

        for fobj in needed:
            try:
                syn.get(
                    fobj["id"],
                    downloadLocation=str(case_dir),
                    ifcollision="keep.local",
                )
                done_files += 1
            except Exception as exc:
                print(f"    ERROR {fobj['name']}: {exc}")
                error_files += 1

        done_cases += 1

    print()
    print("=" * 64)
    print("DOWNLOAD SUMMARY")
    print(f"Cases processed : {done_cases}")
    print(f"Files downloaded: {done_files}")
    print(f"Files skipped   : {skipped_files}")
    print(f"Errors          : {error_files}")
    print(f"Output folder   : {IMAGES_DIR}")
    print("=" * 64)

    return 0 if error_files == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
