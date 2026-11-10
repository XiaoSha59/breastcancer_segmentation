#!/usr/bin/env python3
"""
safety_guard.py
===============
Layer 2 Independent Watchdog Daemon:
Monitors a target training PID every 60 seconds.
If the target PID dies and remains dead for 5 consecutive checks (5 minutes),
it triggers an emergency system shutdown (sudo /sbin/shutdown -h now).
"""

import sys
import os
import time
import subprocess
import datetime

def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def main():
    if len(sys.argv) > 1:
        target_pid = int(sys.argv[1])
    else:
        print("Usage: python3 safety_guard.py <TARGET_PID>")
        sys.exit(1)

    print(f"[{datetime.datetime.now()}] [SafetyGuard] Active. Monitoring PID {target_pid}...", flush=True)
    dead_count = 0

    while True:
        time.sleep(60)
        if not is_pid_alive(target_pid):
            dead_count += 1
            print(f"[{datetime.datetime.now()}] [SafetyGuard] Target PID {target_pid} is NOT alive! Check {dead_count}/5", flush=True)
            if dead_count >= 5:
                print(f"[{datetime.datetime.now()}] [SafetyGuard] Target dead for 5 minutes. Triggering emergency shutdown now!", flush=True)
                subprocess.run(['sudo', '/sbin/shutdown', '-h', 'now'])
                break
        else:
            dead_count = 0

if __name__ == "__main__":
    main()
