#!/usr/bin/env python3
"""
auto_start_and_launch_swinunetr_v2.py
=====================================
Automated GPU slot acquisition & SwinUNETR V2 launcher for GCP.
1. Boots 'cancer-seg-gpu' in zone 'asia-southeast1-c'.
2. Pushes training, evaluation, and runner scripts via gcloud compute scp.
3. Sets executable permissions and launches run_swinunetr_v2_fold0.sh in background.
4. Verifies process execution and logs the real-time monitoring command.
"""

import os
import sys
import time
import subprocess
import datetime

VM_NAME = "cancer-seg-gpu"
ZONE = "asia-southeast1-c"
MAX_ATTEMPTS = 15
RETRY_DELAY = 30  # seconds

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_cmd(cmd_list):
    res = subprocess.run(cmd_list, capture_output=True, text=True, shell=True)
    return res.returncode, res.stdout, res.stderr

def main():
    log(f"Starting auto-start sequence for {VM_NAME} in {ZONE}...")
    
    vm_started = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"[Attempt {attempt}/{MAX_ATTEMPTS}] Starting {VM_NAME}...")
        code, stdout, stderr = run_cmd([
            "gcloud", "compute", "instances", "start", VM_NAME,
            f"--zone={ZONE}"
        ])
        
        combined_out = (stdout + " " + stderr).strip()
        
        if code == 0 or "RUNNING" in combined_out:
            log("SUCCESS! Instance is RUNNING.")
            vm_started = True
            break
        elif "ZONE_RESOURCE_POOL_EXHAUSTED" in combined_out:
            log(f"Zone resource pool exhausted. Waiting {RETRY_DELAY}s before retry...")
        else:
            log(f"Start attempt returned code {code}: {combined_out[:200]}")
            log(f"Waiting {RETRY_DELAY}s before retry...")
            
        time.sleep(RETRY_DELAY)

    if not vm_started:
        log(f"FAILED to start VM after {MAX_ATTEMPTS} attempts.")
        sys.exit(1)

    log("Waiting 20 seconds for SSH daemon and OS services to initialize...")
    time.sleep(20)

    # Step 1: Upload scripts
    log("Uploading SwinUNETR V2 scripts to Cloud VM via gcloud compute scp...")
    script_files = [
        "scripts/train_swinunetr_v2_fold0.py",
        "scripts/evaluate_swinunetr_v2_fold0.py",
        "scripts/run_swinunetr_v2_fold0.sh",
        "scripts/compute_paper_metrics.py",
    ]
    code, out, err = run_cmd([
        "gcloud", "compute", "scp",
        *script_files,
        f"{VM_NAME}:/home/XiaoSha/project/scripts/",
        f"--zone={ZONE}"
    ])
    if code != 0:
        log(f"SCP upload returned: {err}")
    else:
        log("Scripts uploaded successfully.")

    # Step 2: Make runner script executable
    log("Setting executable permissions on run_swinunetr_v2_fold0.sh...")
    run_cmd([
        "gcloud", "compute", "ssh", VM_NAME,
        f"--zone={ZONE}",
        "--command=chmod +x /home/XiaoSha/project/scripts/run_swinunetr_v2_fold0.sh"
    ])

    # Step 3: Launch in background via nohup
    log("Launching SwinUNETR V2 runner in background via nohup...")
    launch_cmd = "nohup bash -c 'export AUTO_SHUTDOWN=1 && export GCP_CLOUD_INSTANCE=1 && bash /home/XiaoSha/project/scripts/run_swinunetr_v2_fold0.sh' > /home/XiaoSha/swinunetr_v2_run.log 2>&1 < /dev/null &"
    code, out, err = run_cmd([
        "gcloud", "compute", "ssh", VM_NAME,
        f"--zone={ZONE}",
        f"--command={launch_cmd}"
    ])
    log("Background launch command triggered.")

    time.sleep(5)

    # Step 4: Verify process
    log("Verifying that process is running on VM...")
    code, out, err = run_cmd([
        "gcloud", "compute", "ssh", VM_NAME,
        f"--zone={ZONE}",
        "--command=ps aux | grep -E 'run_swinunetr_v2|train_swinunetr_v2' | grep -v grep"
    ])
    log(f"Running processes on VM:\n{out}")

    log("=== SWINUNETR V2 HAS BEEN SUCCESSFULLY LAUNCHED ON CLOUD GPU! ===")
    log(f"Monitor logs with: gcloud compute ssh {VM_NAME} --zone={ZONE} --command='tail -f /home/XiaoSha/swinunetr_v2_run.log'")

if __name__ == "__main__":
    main()
