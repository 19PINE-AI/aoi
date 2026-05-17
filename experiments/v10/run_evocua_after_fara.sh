#!/bin/bash
# Wait for Fara eval to finish, then swap GPU memory to EvoCUA-32B.
#
# Polls results/v10_structured_fara.json for >= 100 entries.  When seen,
# stops the Fara vLLM (port 5500), starts EvoCUA-32B vLLM (port 5504),
# and runs the structured eval against it.
set -eo pipefail
cd "$(dirname "$0")/../.."

VLLM=/home/ubuntu/aoi-env/bin/vllm
PY=/home/ubuntu/aoi-env/bin/python
LOG_DIR=$(pwd)/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/evocua_${TS}.log"

echo "Wait for Fara eval to finish (logging to $LOG)" | tee -a "$LOG"

# Poll for Fara completion
while true; do
    if [ -f "results/v10_structured_fara.json" ]; then
        N=$($PY -c "import json; print(len(json.load(open('results/v10_structured_fara.json'))))" 2>/dev/null || echo 0)
        if [ "$N" -ge 100 ]; then
            echo "Fara eval complete ($N results) at $(date)" | tee -a "$LOG"
            break
        fi
        echo "Fara progress: $N/100 at $(date)" >> "$LOG"
    fi
    sleep 60
done

# Stop the Fara vLLM (find and kill it)
echo "Stopping Fara vLLM (port 5500)" | tee -a "$LOG"
FARA_PID=$(ss -tlnp 2>/dev/null | grep ":5500" | grep -oP 'pid=\K\d+' | head -1 || true)
if [ -z "$FARA_PID" ]; then
    # fallback: pgrep
    FARA_PID=$(pgrep -f "vllm.*microsoft/Fara-7B" | head -1 || true)
fi
if [ -n "$FARA_PID" ]; then
    echo "Killing PID $FARA_PID" | tee -a "$LOG"
    kill -INT "$FARA_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "$FARA_PID" 2>/dev/null || break
        sleep 1
    done
    kill -9 "$FARA_PID" 2>/dev/null || true
fi
sleep 15

# Start EvoCUA vLLM on port 5504 with high mem budget.
EVO_LOG="$LOG_DIR/vllm_evocua_${TS}.log"
echo "Starting EvoCUA-32B vLLM (port 5504)" | tee -a "$LOG"
nohup $VLLM serve "meituan/EvoCUA-32B-20260105" \
    --port 5504 \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --trust-remote-code \
    > "$EVO_LOG" 2>&1 &
EVO_PID=$!
echo "EvoCUA vLLM PID $EVO_PID (log $EVO_LOG)" | tee -a "$LOG"

# Wait up to 25 minutes for the 32B model to load
for i in $(seq 1 1500); do
    if curl -sf http://localhost:5504/health >/dev/null 2>&1; then
        echo "EvoCUA vLLM ready after ${i}s" | tee -a "$LOG"
        break
    fi
    if ! kill -0 "$EVO_PID" 2>/dev/null; then
        echo "[FATAL] EvoCUA vLLM died" | tee -a "$LOG"
        tail -30 "$EVO_LOG" >> "$LOG"
        exit 1
    fi
    sleep 1
done

# Run the eval
echo "Running EvoCUA-32B standard_structured eval" | tee -a "$LOG"
VLLM_BASE_URL=http://localhost:5504/v1 $PY -u experiments/v10/run_structured.py \
    --model evocua-32b \
    --out results/v10_structured_evocua.json \
    >> "$LOG" 2>&1
echo "EvoCUA eval done at $(date)" | tee -a "$LOG"
