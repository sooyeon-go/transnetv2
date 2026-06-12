#!/usr/bin/env bash
# TransNet V2 PyTorch inference용 conda 환경 생성 스크립트
# README: inference-pytorch/README.md

set -euo pipefail

ENV_NAME="${1:-transnetv2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.7}"

if ! command -v conda &>/dev/null; then
    echo "Error: conda가 설치되어 있지 않습니다." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Error: conda 환경 '$ENV_NAME'이(가) 이미 존재합니다." >&2
    echo "다른 이름을 쓰려면: $0 <env_name>" >&2
    exit 1
fi

echo "==> conda 환경 생성: $ENV_NAME (python=$PYTHON_VERSION)"
conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> PyTorch 1.7.1 + CUDA 10.1 설치"
conda install pytorch=1.7.1 cudatoolkit=10.1 -c pytorch -y

echo "==> 가중치 변환 및 inference용 pip 패키지 설치"
# TF 2.1은 protobuf 4.x와 호환되지 않음 (convert_weights.py용, inference에는 TF 불필요)
pip install --no-cache-dir "tensorflow==2.1" "protobuf==3.19.6" numpy ffmpeg-python

echo
echo "환경 생성 완료."
echo
echo "사용 방법:"
echo "  conda activate $ENV_NAME"
echo "  cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo
echo "TensorFlow 가중치를 PyTorch로 변환 (최초 1회):"
echo "  python convert_weights.py [--test]"
echo
echo "변환 후 생성되는 파일: transnetv2-pytorch-weights.pth"
echo
echo "비디오 inference (장면 전환 횟수 출력):"
echo "  python transnetv2_infer.py /path/to/video.mp4"
echo "  # stdout: video.mp4: 3"
echo "  # 파일:   video.mp4.transition_count.txt  (내용: 0, 1, 2, ...)"
