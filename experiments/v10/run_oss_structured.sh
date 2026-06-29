#!/bin/bash
# Sequential launcher for Fara-7B and EvoCUA-32B standard_structured evals.
# Boots vLLM, waits for /health, runs the 100-task eval, stops vLLM,
# repeats for the next model.
#
# CUDA works fine despite the NVML mismatch (Whisper large-v3 is already
# serving GPU inference on the same box).
#
# Usage:
#     bash experiments/v10/run_oss_structured.sh
set -eo pipefail
cd "$(dirname "$0")/../.."

VLLM="${VLLM:-vllm}"
PY="${PYTHON:-python}"
LOG_DIR=$(pwd)/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/oss_structured_${TS}.log"
echo "Logging to $MAIN_LOG"

start_vllm() {
    local model=$1
    local port=$2
    local gpu_util=$3
    local vllm_log="$LOG_DIR/vllm_${port}_${TS}.log"
    echo "[$(date)] Starting vLLM $model on :$port (gpu_util=$gpu_util)" | tee -a "$MAIN_LOG"
    # nohup so the server survives this shell
    nohup $VLLM serve "$model" \
        --port "$port" \
        --dtype auto \
        --max-model-len 4096 \
        --gpu-memory-utilization "$gpu_util" \
        --trust-remote-code \
        > "$vllm_log" 2>&1 &
    local pid=$!
    echo "vLLM PID: $pid (log: $vllm_log)" | tee -a "$MAIN_LOG"
    echo "$pid"
}

wait_vllm() {
    local port=$1
    local timeout=$2
    for i in $(seq 1 "$timeout"); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "[$(date)] vLLM :$port ready after ${i}s" | tee -a "$MAIN_LOG"
            return 0
        fi
        sleep 1
    done
    echo "[FATAL] vLLM :$port did not start within ${timeout}s" | tee -a "$MAIN_LOG"
    return 1
}

stop_vllm() {
    local pid=$1
    echo "[$(date)] Stopping vLLM PID $pid" | tee -a "$MAIN_LOG"
    kill -INT "$pid" 2>/dev/null || true
    # Wait for graceful shutdown so GPU memory is freed before next launch
    for _ in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    sleep 5
}

# ────────────────────────────────────────────────────────────────────
# Stage 1: Fara-7B
# ────────────────────────────────────────────────────────────────────
echo "===== Fara-7B standard_structured =====" | tee -a "$MAIN_LOG"
# Port 5000 is held by an unrelated Docker container.  Use 5500.
# 7B FP16 weights ~14GB; 0.30 of 102GB = 30GB total budget. Comfortable.
F_PID=$(start_vllm "microsoft/Fara-7B" 5500 "0.30")
if wait_vllm 5500 600; then
    VLLM_BASE_URL=http://localhost:5500/v1 $PY -u experiments/v10/run_structured.py \
        --model fara-7b \
        --out results/v10_structured_fara.json \
        >> "$MAIN_LOG" 2>&1
else
    echo "[FATAL] Skipping Fara-7B eval (vLLM did not start)" | tee -a "$MAIN_LOG"
fi
stop_vllm "$F_PID"

# ────────────────────────────────────────────────────────────────────
# Stage 2: EvoCUA-32B
# ────────────────────────────────────────────────────────────────────
echo "===== EvoCUA-32B standard_structured =====" | tee -a "$MAIN_LOG"
# 32B FP16 weights ~64GB; Whisper holds ~10GB. 0.80 of 102GB = 82GB budget,
# leaving ~10GB for the Whisper service that other concurrent evals depend on.
E_PID=$(start_vllm "meituan/EvoCUA-32B-20260105" 5504 "0.80")
if wait_vllm 5504 1500; then
    VLLM_BASE_URL=http://localhost:5504/v1 $PY -u experiments/v10/run_structured.py \
        --model evocua-32b \
        --out results/v10_structured_evocua.json \
        >> "$MAIN_LOG" 2>&1
else
    echo "[FATAL] Skipping EvoCUA-32B eval (vLLM did not start)" | tee -a "$MAIN_LOG"
fi
stop_vllm "$E_PID"

echo "===== DONE $(date) =====" | tee -a "$MAIN_LOG"
