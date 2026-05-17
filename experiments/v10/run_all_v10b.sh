#!/bin/bash
# Sequential launcher for v10b background runs:
#   1. Claude AOI-full seed 2 (100 tasks)
#   2. Claude AOI-full seed 3 (100 tasks)
#   3. OpenAI gpt-realtime-2.0 sanity + audio subset
#
# Grok-4 is intentionally NOT launched here because XAI_API_KEY is unset.
# To add it later:
#   XAI_API_KEY=... python experiments/v10/run_grok_main.py
#
# Sequential, not parallel: the PulseAudio virtual devices and Whisper
# service are shared infrastructure that have shown leakage under concurrent
# evals (see project notes on the v4 concurrent-AOI bug).
set -eo pipefail
cd "$(dirname "$0")/../.."

LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/v10b_${TS}.log"

echo "Logging to $LOG"
echo "Started $(date)" >> "$LOG"

# Ensure PulseAudio virtual devices are loaded
python3 -c "from aoi.audio_pipeline import PulseAudioManager; assert PulseAudioManager.ensure_devices()" >> "$LOG" 2>&1

# Ensure Whisper service is live
if ! curl -sf http://localhost:8786/health > /dev/null; then
  echo "[FATAL] Whisper service is not responding on :8786" >> "$LOG"
  exit 1
fi

echo "----- variance seed 2 -----" >> "$LOG"
/home/ubuntu/aoi-env/bin/python -u experiments/v10/run_variance.py --seeds 2 >> "$LOG" 2>&1

echo "----- variance seed 3 -----" >> "$LOG"
/home/ubuntu/aoi-env/bin/python -u experiments/v10/run_variance.py --seeds 3 >> "$LOG" 2>&1

echo "----- gpt-realtime-2.0 -----" >> "$LOG"
/home/ubuntu/aoi-env/bin/python -u experiments/v10/run_realtime_v2.py >> "$LOG" 2>&1

echo "----- done $(date) -----" >> "$LOG"
