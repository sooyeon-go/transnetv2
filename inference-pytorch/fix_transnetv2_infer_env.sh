#!/usr/bin/env bash
# transnetv2-infer 환경에서 torch import 오류(iJIT_NotifyEvent 등) 복구

set -euo pipefail

ENV_NAME="${1:-transnetv2-infer}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

if ! command -v conda &>/dev/null; then
    echo "Error: conda가 설치되어 있지 않습니다." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Error: conda 환경 '$ENV_NAME'이(가) 없습니다." >&2
    echo "새로 만들려면: bash setup_conda_env_infer.sh" >&2
    exit 1
fi

conda activate "$ENV_NAME"

echo "==> 깨진 conda PyTorch 제거 (있으면)"
conda remove -y pytorch torchvision torchaudio pytorch-cuda pytorch-mutex cuda-cudart cuda-version 2>/dev/null || true
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "==> pip wheel로 PyTorch 재설치"
pip install --no-cache-dir torch torchvision torchaudio --index-url "${TORCH_INDEX}"

echo "==> inference 패키지 확인"
pip install --no-cache-dir "numpy<2.3" ffmpeg-python

echo "==> import 테스트"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    idx = 5 if torch.cuda.device_count() > 5 else 0
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    print(f"test GPU {idx}: {name} (sm_{cap[0]}{cap[1]})")
    torch.zeros(1, device=f"cuda:{idx}")
    print("cuda tensor test: OK")
PY

echo
echo "복구 완료. 이제:"
echo "  conda activate $ENV_NAME"
echo "  ./run_enrich_json.sh"
