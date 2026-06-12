#!/usr/bin/env bash
# TransNet V2 inference 전용 환경 (H200/H100 등 최신 GPU용)
# 가중치 변환은 이미 끝났다면 TF 설치 불필요

set -euo pipefail

ENV_NAME="${1:-transnetv2-infer}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda &>/dev/null; then
    echo "Error: conda가 설치되어 있지 않습니다." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Error: conda 환경 '$ENV_NAME'이(가) 이미 존재합니다." >&2
    exit 1
fi

echo "==> conda 환경 생성: $ENV_NAME (python=$PYTHON_VERSION)"
conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> PyTorch 2.x + CUDA 12 설치 (H200 sm_90 지원)"
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

echo "==> inference용 pip 패키지 설치"
pip install --no-cache-dir numpy ffmpeg-python

echo
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {name} (sm_{cap[0]}{cap[1]})")
    x = torch.zeros(1, device="cuda:0")
    print("cuda tensor test: OK")
PY

echo
echo "환경 생성 완료."
echo "  conda activate $ENV_NAME"
echo "  cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "  ./run_enrich_json.sh"
