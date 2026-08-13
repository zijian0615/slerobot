#!/usr/bin/env bash
# Sync and run SAM2 union task-mask extraction on remote GPU.
# Environment variables (RTX4090-prefixed to avoid clashing with other hosts):
#   RTX4090_SSHPASS   - password for sshpass (optional)
#   RTX4090_REMOTE    - default liang@10.56.147.233
#   RTX4090_REMOTE_DIR - default ~/slerobot_sam2
#
# Example:
#   export RTX4090_SSHPASS='...'
#   EPISODE=20 ./scripts/run_sam2_task_mask_remote.sh
set -euo pipefail

REMOTE="${RTX4090_REMOTE:-liang@10.56.147.233}"
REMOTE_DIR="${RTX4090_REMOTE_DIR:-~/slerobot_sam2}"
if [[ -n "${RTX4090_SSHPASS:-}" ]]; then
  export SSHPASS="${RTX4090_SSHPASS}"
elif [[ -n "${SSHPASS:-}" ]]; then
  export SSHPASS
fi
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

EPISODE="${EPISODE:-20}"          # trigger episode in xa3_clean123_bad2
CAMERA="${CAMERA:-side}"
REPO_ID="${REPO_ID:-zijian2022/xa3_clean123_bad2}"
OUTPUT_NAME="${OUTPUT_NAME:-M_task_side_ep${EPISODE}_trigger.npy}"
LOCAL_FRAMES_DIR="${LOCAL_ROOT}/outputs/sam2_task_mask/ep${EPISODE}_frames"
REMOTE_FRAMES_DIR="${REMOTE_DIR}/outputs/sam2_task_mask/ep${EPISODE}_frames"
PYTHON_LOCAL="${PYTHON_LOCAL:-/Users/zhangzijian/anaconda3/envs/slerobot/bin/python}"

run_ssh() {
  if [[ -n "${SSHPASS:-}" ]]; then
    sshpass -e ssh -o StrictHostKeyChecking=accept-new "${REMOTE}" "$@"
  else
    ssh "${REMOTE}" "$@"
  fi
}

run_rsync() {
  if [[ -n "${SSHPASS:-}" ]]; then
    sshpass -e rsync -e "ssh -o StrictHostKeyChecking=accept-new" "$@"
  else
    rsync "$@"
  fi
}

echo "==> Prepare remote dir ${REMOTE}:${REMOTE_DIR}"
run_ssh "mkdir -p ${REMOTE_DIR}/src ${REMOTE_DIR}/outputs/sam2_task_mask ${REMOTE_DIR}/outputs/sam2_checkpoints"

echo "==> Sync code + boxes + checkpoint"
run_rsync -avz \
  --exclude '__pycache__' \
  "${LOCAL_ROOT}/src/" "${REMOTE}:${REMOTE_DIR}/src/"
run_rsync -avz \
  "${LOCAL_ROOT}/pyproject.toml" \
  "${REMOTE}:${REMOTE_DIR}/"
run_rsync -avz \
  "${LOCAL_ROOT}/outputs/sam2_task_mask/boxes_side.json" \
  "${REMOTE}:${REMOTE_DIR}/outputs/sam2_task_mask/boxes_side.json"
run_rsync -avz \
  "${LOCAL_ROOT}/outputs/sam2_checkpoints/sam2.1_hiera_tiny.pt" \
  "${REMOTE}:${REMOTE_DIR}/outputs/sam2_checkpoints/sam2.1_hiera_tiny.pt"

echo "==> Export episode ${EPISODE} frames locally (${CAMERA})"
mkdir -p "${LOCAL_FRAMES_DIR}"
PYTHONPATH="${LOCAL_ROOT}/src" "${PYTHON_LOCAL}" - <<PY
from pathlib import Path
import cv2
import numpy as np
from slerobot.datasets.slerobot_datasets import sLerobotDataset

def tensor_to_bgr(t):
    arr = t.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

repo_id = "${REPO_ID}"
episode = ${EPISODE}
camera = "${CAMERA}"
frames_dir = Path("${LOCAL_FRAMES_DIR}")
dataset = sLerobotDataset(repo_id, episodes=[episode], download_videos=True)
cam_key = next(k for k in dataset.meta.features if "images" in k and (k == camera or k.split(".")[-1] == camera))
episode_indices = {}
for idx in range(len(dataset)):
    ep = dataset.hf_dataset[idx]["episode_index"].item()
    episode_indices.setdefault(ep, []).append(idx)
indices = episode_indices[episode]
frames_dir.mkdir(parents=True, exist_ok=True)
for frame_idx, dataset_index in enumerate(indices):
    bgr = tensor_to_bgr(dataset[dataset_index][cam_key])
    cv2.imwrite(str(frames_dir / f"{frame_idx:05d}.jpg"), bgr)
print(f"exported {len(indices)} frames to {frames_dir}")
PY

echo "==> Sync frames to remote"
run_rsync -avz \
  "${LOCAL_FRAMES_DIR}/" \
  "${REMOTE}:${REMOTE_FRAMES_DIR}/"

echo "==> Run SAM2 on remote GPU (episode ${EPISODE}, trigger)"
run_ssh bash -s <<EOF
set -euo pipefail
cd ${REMOTE_DIR}

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install opencv-python-headless hydra-core einops

if [ ! -d third_party/sam2 ]; then
  mkdir -p third_party
  git clone https://github.com/facebookresearch/sam2.git third_party/sam2
fi
python -m pip install -q -e third_party/sam2

rm -rf sam2
mkdir -p run
cd run

export PYTHONPATH="\${PWD}/../src:\${PYTHONPATH:-}"
python ../src/slerobot/scripts/slerobot_build_sam2_task_mask.py \
  --reuse_boxes \
  --boxes_path ../outputs/sam2_task_mask/boxes_side.json \
  --frames_dir ../outputs/sam2_task_mask/ep${EPISODE}_frames \
  --sam2_checkpoint ../outputs/sam2_checkpoints/sam2.1_hiera_tiny.pt \
  --output_path ../outputs/sam2_task_mask/${OUTPUT_NAME} \
  --device cuda
EOF

echo "==> Fetch results"
mkdir -p "${LOCAL_ROOT}/outputs/sam2_task_mask"
run_rsync -avz \
  "${REMOTE}:${REMOTE_DIR}/outputs/sam2_task_mask/${OUTPUT_NAME}" \
  "${REMOTE}:${REMOTE_DIR}/outputs/sam2_task_mask/${OUTPUT_NAME%.npy}.preview.jpg" \
  "${LOCAL_ROOT}/outputs/sam2_task_mask/"

echo "Done:"
echo "  ${LOCAL_ROOT}/outputs/sam2_task_mask/${OUTPUT_NAME}"
echo "  ${LOCAL_ROOT}/outputs/sam2_task_mask/${OUTPUT_NAME%.npy}.preview.jpg"
