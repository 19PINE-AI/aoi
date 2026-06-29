#!/bin/bash
# Serve open-source CU models via vLLM for evaluation.
#
# Usage:
#   ./experiments/serve_local_model.sh fara-7b      # Port 5000
#   ./experiments/serve_local_model.sh ui-tars-7b   # Port 5001
#
# The model is served with an OpenAI-compatible API endpoint.
# Run evaluations with: python experiments/run_full_eval.py --phase local

set -e

MODEL=${1:-fara-7b}

case $MODEL in
    fara-7b)
        HF_ID="microsoft/Fara-7B"
        PORT=5000
        EXTRA_ARGS="--max-model-len 4096"
        ;;
    ui-tars-7b)
        HF_ID="ByteDance-Seed/UI-TARS-1.5-7B"
        PORT=5001
        EXTRA_ARGS="--max-model-len 4096"
        ;;
    ui-tars-72b)
        HF_ID="ByteDance-Seed/UI-TARS-72B-DPO"
        PORT=5002
        EXTRA_ARGS="--max-model-len 4096 --tensor-parallel-size 1"
        ;;
    opencua-7b)
        HF_ID="xlangai/OpenCUA-7B"
        PORT=5003
        EXTRA_ARGS="--max-model-len 4096 --trust-remote-code"
        ;;
    evocua-32b)
        HF_ID="meituan/EvoCUA-32B-20260105"
        PORT=5004
        EXTRA_ARGS="--max-model-len 4096"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available: fara-7b, ui-tars-7b, ui-tars-72b, opencua-7b, evocua-32b"
        exit 1
        ;;
esac

echo "Serving $HF_ID on port $PORT..."
"${PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
    --model "$HF_ID" \
    --port $PORT \
    --host 0.0.0.0 \
    --dtype float16 \
    $EXTRA_ARGS
