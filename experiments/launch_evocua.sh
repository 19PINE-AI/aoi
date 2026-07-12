#!/bin/bash
# Launch EvoCUA-32B experiments
# Run this after: (1) EvoCUA download completes, (2) Fara-7B evals finish, (3) Fara-7B vLLM stopped

set -e

source "${VENV:-$HOME/aoi-env}/bin/activate"

echo "=== Stopping Fara-7B vLLM server ==="
pkill -f "vllm serve microsoft/Fara-7B" || true
sleep 5

echo "=== Starting EvoCUA-32B vLLM server on port 5004 ==="
# Use AWQ quantization if available, otherwise auto dtype with reduced context
vllm serve meituan/EvoCUA-32B-20260105 \
    --port 5004 \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM to load model..."
for i in $(seq 1 120); do
    if curl -s http://localhost:5004/health 2>/dev/null | grep -q "ok\|healthy"; then
        echo "vLLM ready after ${i}s"
        break
    fi
    sleep 5
done

TASKS="A-E1 A-E2 A-E3 A-M1 A-M2 A-M3 A-M4 A-H1 A-H2 A-H3 B-E1 B-E2 B-E3 B-M1 B-M2 B-M3 B-M4 B-H1 B-H2 B-H3 C-E1 C-E2 C-E3 C-M1 C-M2 C-M3 C-M4 C-H1 C-H2 C-H3 D-E1 D-E2 D-E3 D-M1 D-M2 D-M3 D-M4 D-H1 D-H2 D-H3 E-E1 E-E2 E-E3 E-M1 E-M2 E-M3 E-M4 E-H1 E-H2 E-H3 F-E1 F-E2 F-E3 F-M1 F-M2 F-M3 F-M4 F-H1 F-H2 F-H3 G-E1 G-E2 G-E3 G-M1 G-M2 G-M3 G-M4 G-H1 G-H2 G-H3 H-E1 H-E2 H-E3 H-M1 H-M2 H-M3 H-M4 H-H1 H-H2 H-H3 I-E1 I-E2 I-E3 I-M1 I-M2 I-M3 I-M4 I-H1 I-H2 I-H3 J-E1 J-E2 J-E3 J-M1 J-M2 J-M3 J-M4 J-H1 J-H2 J-H3"

echo "=== Running EvoCUA-32B + standard ==="
python experiments/run_10task_eval.py \
    --model evocua-32b \
    --mode standard \
    --max-steps 15 \
    --tasks $TASKS \
    --output results/v9_full_100_evocua32b_standard.json \
    2>&1 | tee /tmp/v9_evocua_standard_eval.log

echo "=== Running EvoCUA-32B + aoi_full ==="
python experiments/run_10task_eval.py \
    --model evocua-32b \
    --mode aoi_full \
    --max-steps 15 \
    --tasks $TASKS \
    --output results/v9_full_100_evocua32b_aoi.json \
    2>&1 | tee /tmp/v9_evocua_aoi_eval.log

echo "=== EvoCUA-32B experiments complete ==="
