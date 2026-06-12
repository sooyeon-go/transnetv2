#!/usr/bin/env bash
# OpenVid optical-flow JSON에 transition_count 추가

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="${SCRIPT_DIR}/transnetv2-pytorch-weights.pth"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/enrich_latest.log"

if [[ ! -f "$WEIGHTS" ]]; then
    echo "Error: weights not found: $WEIGHTS" >&2
    echo "Run: python ${SCRIPT_DIR}/convert_weights.py" >&2
    exit 1
fi

echo "Checking GPU/PyTorch compatibility..."
export SCRIPT_DIR
if ! python - <<'PY'
import os
import sys
import torch

script_dir = os.environ["SCRIPT_DIR"]

if not torch.cuda.is_available():
    print("WARNING: CUDA not available, will run on CPU (very slow).", file=sys.stderr)
    sys.exit(0)

idx = 5 if torch.cuda.device_count() > 5 else 0
name = torch.cuda.get_device_name(idx)
cap = torch.cuda.get_device_capability(idx)
print(f"PyTorch {torch.__version__}, test GPU {idx}: {name} (sm_{cap[0]}{cap[1]})")
try:
    torch.zeros(1, device=f"cuda:{idx}")
except Exception as exc:
    print(
        f"ERROR: GPU {idx} not usable with PyTorch {torch.__version__}: {exc}\n"
        "H200/H100 needs PyTorch 2.x. Run:\n"
        f"  bash {script_dir}/setup_conda_env_infer.sh\n"
        "  conda activate transnetv2-infer\n"
        f"  bash {script_dir}/run_enrich_json.sh",
        file=sys.stderr,
    )
    sys.exit(1)
print("GPU check: OK")
PY
then
    exit 1
fi

echo "Logging to ${LOG}"
echo "Check progress/time: python ${SCRIPT_DIR}/check_enrich_progress.py"
echo "Timing summary:      ${LOG_DIR}/enrich_timing_summary.json"
echo "Or: tail -f ${LOG}"

python "${SCRIPT_DIR}/enrich_json_transitions.py" \
    --use-default-jsons \
    --weights "$WEIGHTS" \
    --gpus 5,6,7 \
    --inplace \
    --resume \
    --save-every 100 \
    "$@" 2>&1 | tee "$LOG"
