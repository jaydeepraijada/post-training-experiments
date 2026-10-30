#!/usr/bin/env bash
# Run GRPO on a shared machine: stop this job (and nothing else) if free system
# memory drops below MIN_AVAILABLE_MB, so it can't push other services into OOM.
#
# Usage: ./run_guarded.sh <config.yaml> <output_model_id>
set -uo pipefail

CONFIG="$1"
OUTPUT_ID="$2"
MIN_AVAILABLE_MB="${MIN_AVAILABLE_MB:-3000}"
LOG_DIR="models/$OUTPUT_ID"
mkdir -p "$LOG_DIR"

.venv/bin/python -u reasoning/grpo/train.py "$CONFIG" -o "$OUTPUT_ID" >> "$LOG_DIR/train.log" 2>&1 &
TRAIN_PID=$!

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    AVAILABLE_MB=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    if [ "$AVAILABLE_MB" -lt "$MIN_AVAILABLE_MB" ]; then
        echo "$(date -Is) MemAvailable=${AVAILABLE_MB}MB < ${MIN_AVAILABLE_MB}MB, stopping pid $TRAIN_PID" | tee -a "$LOG_DIR/train.log"
        kill "$TRAIN_PID"
        sleep 20
        kill -9 "$TRAIN_PID" 2>/dev/null
        break
    fi
    sleep 5
done

wait "$TRAIN_PID"
STATUS=$?
echo "$(date -Is) train.py exited with status $STATUS" | tee -a "$LOG_DIR/train.log"
exit "$STATUS"
