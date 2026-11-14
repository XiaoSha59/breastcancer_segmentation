#!/usr/bin/env bash
TARGET_PID=1622
LOG_FILE="/home/XiaoSha/auto_shutdown_eval.log"

echo "[$(date)] Auto-shutdown watchdog started for eval PID $TARGET_PID..." >> "$LOG_FILE"

while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 20
done

echo "[$(date)] Evaluation PID $TARGET_PID finished! Syncing files..." >> "$LOG_FILE"
sync

echo "[$(date)] Waiting 120 seconds before safe poweroff to save GCP credits..." >> "$LOG_FILE"
sleep 120

echo "[$(date)] Executing sudo /sbin/shutdown -h now..." >> "$LOG_FILE"
sudo /sbin/shutdown -h now
