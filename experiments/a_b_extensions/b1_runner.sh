#!/usr/bin/env bash
# B1: Run open-source CU model replication.
# Sequentially runs Qwen3-VL-235B and Qwen3-VL-30B-A3B (both via OpenRouter)
# in standard + AOI-full configurations on the full 100-task DynaCU-Bench.
#
# Trigger: Should only be launched AFTER A1 (Gemini 3 keyframe probe) has
# finished, since both use the PulseAudio virtual_speaker pipeline and
# can't share it cleanly.

set -u
cd "$(dirname "$0")/../.."

export VLLM_API_KEY=${OPENROUTER_API_KEY?must be set}
mkdir -p logs/extensions

# Larger MoE first (more interesting datapoint).
for spec in \
    "qwen3-vl-235b-or qwen3-vl-235b" \
    "qwen3-vl-30b-or  qwen3-vl-30b" \
    ; do
    set -- $spec
    model=$1
    tag=$2
    echo "============================="
    echo " B1: $model  (tag=$tag)"
    echo "============================="
    for mode in standard aoi_full; do
        python3 experiments/a_b_extensions/b1_open_source_replication.py \
            --model "$model" --tag "$tag" --modes "$mode" \
            >> logs/extensions/b1_${tag}_${mode}.log 2>&1
        rc=$?
        echo "  $mode finished (rc=$rc)"
    done
done

echo "All B1 done."
