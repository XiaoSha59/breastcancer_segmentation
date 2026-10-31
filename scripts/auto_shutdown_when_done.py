"""Watchdog to automatically power off GCP VM once downloads finish.

Saves GPU compute costs by shutting down the instance to TERMINATED state.
"""

from __future__ import annotations

import datetime
import subprocess
import time
from pathlib import Path

LOG_DIR = Path("/home/XiaoSha")
STATUS_FILE = LOG_DIR / "download_finished.txt"


def is_running(script_name: str) -> bool:
    res = subprocess.run(["pgrep", "-f", script_name], capture_output=True, text=True)
    # Filter out current watcher process if matching
    pids = [p.strip() for p in res.stdout.strip().split() if p.strip()]
    return len(pids) > 0


def main() -> None:
    print(f"[{datetime.datetime.now()}] Auto-shutdown watcher started.", flush=True)

    # Allow scripts to initialize
    time.sleep(30)

    while True:
        ispy1_active = is_running("download_ispy1_vm.py")
        ispy2_active = is_running("download_ispy2_vm.py")

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not ispy1_active and not ispy2_active:
            print(f"[{now_str}] Both downloads finished! Preparing auto-shutdown...", flush=True)
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                f.write(f"All downloads finished at {now_str}\n")

            # Sync filesystem caches
            subprocess.run(["sync"])
            time.sleep(5)

            print(f"[{now_str}] Executing sudo shutdown -h now...", flush=True)
            subprocess.run(["sudo", "shutdown", "-h", "now"])
            break
        else:
            time.sleep(60)


if __name__ == "__main__":
    main()
