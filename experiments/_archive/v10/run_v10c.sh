#!/bin/bash
# v10c sequential launcher:
#   1. Gemini 3 Flash (sidebar to the v9 Gemini 2.5 result)
#   2. Grok-4.3 (latest Grok)
#   3. Grok-4-fast-reasoning (no-latency-confound)
#
# Sequential to avoid PulseAudio + browser env contention.
set -eo pipefail
cd "$(dirname "$0")/../.."

PY="${PYTHON:-python}"
LOG_DIR=$(pwd)/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/v10c_${TS}.log"
echo "Logging to $LOG"
echo "Started $(date)" >> "$LOG"

# Required for Grok runs
export XAI_API_KEY="${XAI_API_KEY:-}"

# Ensure PulseAudio + Whisper are live
$PY -c "from aoi.audio_pipeline import PulseAudioManager; assert PulseAudioManager.ensure_devices()" >> "$LOG" 2>&1
if ! curl -sf http://localhost:8786/health > /dev/null; then
  echo "[FATAL] Whisper service is not responding on :8786" >> "$LOG"
  exit 1
fi

echo "----- Gemini 3 Flash -----" >> "$LOG"
$PY -u experiments/v10/run_any_main.py --model gemini-3-flash --tag g3flash >> "$LOG" 2>&1

echo "----- Grok-4.3 -----" >> "$LOG"
$PY -u experiments/v10/run_any_main.py --model grok-4.3 --tag grok43 >> "$LOG" 2>&1

echo "----- Grok-4-fast-reasoning -----" >> "$LOG"
$PY -u experiments/v10/run_any_main.py --model grok-4-fast-reasoning --tag grok4fast >> "$LOG" 2>&1

echo "----- done $(date) -----" >> "$LOG"
