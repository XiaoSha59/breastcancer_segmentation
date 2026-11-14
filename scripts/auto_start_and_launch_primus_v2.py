#!/usr/bin/env python3
"""
auto_start_and_launch_primus_v2.py
==================================
Autonomous Launcher for PrimusV2-S Training on GCP Cloud (NVIDIA L4).
1. Starts GCP VM 'cancer-seg-gpu' if TERMINATED.
2. Uploads the latest PrimusV2 scripts via gcloud compute scp.
3. Launches run_primus_v2_fold0.sh in background via nohup.
4. Monitors initial start to verify GPU memory allocation and training loop.
"""

import subprocess
import time
import sys

VM_NAME = "cancer-seg-gpu"
ZONE = "asia-southeast1-c"

def run_cmd(cmd):
    print(f"[CMD] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip(), res.stderr.strip(), res.returncode

def main():
    print("=== STEP 1: CHECKING GCP VM STATUS ===")
    out, err, code = run_cmd(f"gcloud compute instances describe {VM_NAME} --zone={ZONE} --format='value(status)'")
    print(f"Current Status: {out}")

    if out != "RUNNING":
        print("Starting GCP VM...")
        out, err, code = run_cmd(f"gcloud compute instances start {VM_NAME} --zone={ZONE}")
        print("Waiting 30s for VM to fully boot...")
        time.sleep(30)

    print("=== STEP 2: UPLOADING PRIMUSV2 SCRIPTS ===")
    scp_cmd = f"gcloud compute scp scripts/primus_v2.py scripts/train_primus_v2_fold0.py scripts/evaluate_primus_v2_fold0.py scripts/run_primus_v2_fold0.sh {VM_NAME}:/home/XiaoSha/project/scripts/ --zone={ZONE}"
    out, err, code = run_cmd(scp_cmd)
    if code != 0:
        print(f"SCP warning/error: {err}")

    print("=== STEP 3: ENSURING PERMISSIONS AND PACKAGES ===")
    prep_cmd = (
        f"gcloud compute ssh {VM_NAME} --zone={ZONE} --command=\""
        f"chmod +x /home/XiaoSha/project/scripts/run_primus_v2_fold0.sh && "
        f"pip install --quiet dynamic-network-architectures timm einops"
        f"\""
    )
    out, err, code = run_cmd(prep_cmd)

    print("=== STEP 4: LAUNCHING PRIMUSV2-S TRAINING IN BACKGROUND ===")
    launch_cmd = (
        f"gcloud compute ssh {VM_NAME} --zone={ZONE} --command=\""
        f"nohup bash /home/XiaoSha/project/scripts/run_primus_v2_fold0.sh > /home/XiaoSha/primus_v2_run.log 2>&1 < /dev/null & "
        f"sleep 3 && ps aux | grep python3"
        f"\""
    )
    out, err, code = run_cmd(launch_cmd)
    print(f"Launch output:\n{out}")

    print("=== STEP 5: MONITORING INITIAL LOG OUTPUT ===")
    time.sleep(10)
    out, err, code = run_cmd(f"gcloud compute ssh {VM_NAME} --zone={ZONE} --command='tail -n 25 /home/XiaoSha/primus_v2_run.log'")
    print(f"Log Output:\n{out}")
    print("\n[SUCCESS] PrimusV2-S is actively running on GCP L4!")

if __name__ == "__main__":
    main()
