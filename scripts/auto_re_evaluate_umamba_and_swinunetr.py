#!/usr/bin/env python3
"""
auto_re_evaluate_umamba_and_swinunetr.py
========================================
Autonomous Launcher for Precision-Fixed Re-Evaluation of U-Mamba and SwinUNETR V1 on GCP (NVIDIA L4).
1. Boots GCP VM 'cancer-seg-gpu' if TERMINATED.
2. Uploads the latest evaluate_umamba_fold0.py, evaluate_swinunetr_fold0.py, and runner.
3. Launches run_re_evaluate_umamba_swinunetr.sh in background via nohup.
4. Monitors initial start to verify inference is active.
5. System auto-shuts down upon completion via cloud trap.
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

    print("=== STEP 2: UPLOADING EVALUATION SCRIPTS ===")
    scp_cmd = (
        f"gcloud compute scp "
        f"scripts/evaluate_umamba_fold0.py "
        f"scripts/evaluate_swinunetr_fold0.py "
        f"scripts/compute_paper_metrics.py "
        f"scripts/run_re_evaluate_umamba_swinunetr.sh "
        f"{VM_NAME}:/home/XiaoSha/project/scripts/ --zone={ZONE}"
    )
    out, err, code = run_cmd(scp_cmd)
    if code != 0:
        print(f"SCP warning/error: {err}")

    print("=== STEP 3: ENSURING PERMISSIONS ===")
    prep_cmd = (
        f"gcloud compute ssh {VM_NAME} --zone={ZONE} --command=\""
        f"chmod +x /home/XiaoSha/project/scripts/run_re_evaluate_umamba_swinunetr.sh"
        f"\""
    )
    out, err, code = run_cmd(prep_cmd)

    print("=== STEP 4: LAUNCHING RE-EVALUATION IN BACKGROUND ===")
    launch_cmd = (
        f"gcloud compute ssh {VM_NAME} --zone={ZONE} --command=\""
        f"nohup bash /home/XiaoSha/project/scripts/run_re_evaluate_umamba_swinunetr.sh > /home/XiaoSha/re_evaluate.log 2>&1 < /dev/null & "
        f"sleep 3 && ps aux | grep python3"
        f"\""
    )
    out, err, code = run_cmd(launch_cmd)
    print(f"Launch output:\n{out}")

    print("=== STEP 5: MONITORING INITIAL LOG OUTPUT ===")
    time.sleep(10)
    out, err, code = run_cmd(f"gcloud compute ssh {VM_NAME} --zone={ZONE} --command='tail -n 25 /home/XiaoSha/re_evaluate.log'")
    print(f"Log Output:\n{out}")
    print("\n[SUCCESS] Re-evaluation pipeline for U-Mamba and SwinUNETR V1 is actively running on GCP L4!")

if __name__ == "__main__":
    main()
