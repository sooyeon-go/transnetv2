#!/usr/bin/env bash
# OpenVid optical-flow JSON에 transition_count 추가

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="${SCRIPT_DIR}/transnetv2-pytorch-weights.pth"

if [[ ! -f "$WEIGHTS" ]]; then
    echo "Error: weights not found: $WEIGHTS" >&2
    echo "Run: python ${SCRIPT_DIR}/convert_weights.py" >&2
    exit 1
fi

python "${SCRIPT_DIR}/enrich_json_transitions.py" \
    --use-default-jsons \
    --weights "$WEIGHTS" \
    --gpus 5,6,7 \
    --inplace \
    --resume \
    --save-every 100 \
    "$@"
