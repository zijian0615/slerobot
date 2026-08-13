#!/usr/bin/env bash
# Export per-frame SAM2 segmentation overlay video on remote GPU.
set -euo pipefail

REMOTE="${RTX4090_REMOTE:-liang@10.56.147.233}"
REMOTE_DIR="${RTX4090_REMOTE_DIR:-~/slerobot_sam2}"
if [[ -n "${RTX4090_SSHPASS:-}" ]]; then
  export SSHPASS="${RTX4090_SSHPASS}"
elif [[ -n "${SSHPASS:-}" ]]; then
  export SSHPASS
fi
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

EPISODE="${EPISODE:-20}"
FPS="${FPS:-30}"
VIDEO_NAME="${VIDEO_NAME:-segmentation_ep${EPISODE}_side.mp4}"
LOCAL_FRAMES_DIR="${LOCAL_ROOT}/outputs/sam2_task_mask/ep${EPISODE}_frames"

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

echo "==> Sync script + boxes + frames"
run_ssh "mkdir -p ${REMOTE_DIR}/src ${REMOTE_DIR}/outputs/sam2_task_mask/ep${EPISODE}_frames ${REMOTE_DIR}/outputs/sam2_checkpoints"
run_rsync -avz \
  --exclude '__pycache__' \
  "${LOCAL_ROOT}/src/slerobot/scripts/slerobot_build_sam2_task_mask.py" \
  "${REMOTE}:${REMOTE_DIR}/src/slerobot/scripts/slerobot_build_sam2_task_mask.py"
run_rsync -avz \
  "${LOCAL_ROOT}/outputs/sam2_task_mask/boxes_side.json" \
  "${REMOTE}:${REMOTE_DIR}/outputs/sam2_task_mask/boxes_side.json"
run_rsync -avz \
  "${LOCAL_FRAMES_DIR}/" \
  "${REMOTE}:${REMOTE_DIR}/outputs/sam2_task_mask/ep${EPISODE}_frames/"

echo "==> Render segmentation video on remote GPU"
run_ssh bash -s <<EOF
set -euo pipefail
cd ${REMOTE_DIR}
source .venv/bin/activate
rm -rf sam2
mkdir -p run
cd run
export PYTHONPATH="\${PWD}/../src:\${PYTHONPATH:-}"
python ../src/slerobot/scripts/slerobot_build_sam2_task_mask.py \
  --reuse_boxes \
  --boxes_path ../outputs/sam2_task_mask/boxes_side.json \
  --frames_dir ../outputs/sam2_task_mask/ep${EPISODE}_frames \
  --sam2_checkpoint ../outputs/sam2_checkpoints/sam2.1_hiera_tiny.pt \
  --output_path ../outputs/sam2_task_mask/M_task_side_ep${EPISODE}_trigger.npy \
  --segmentation_video_path ../outputs/sam2_task_mask/${VIDEO_NAME} \
  --fps ${FPS} \
  --device cuda
EOF

echo "==> Fetch video"
mkdir -p "${LOCAL_ROOT}/outputs/sam2_task_mask"
run_rsync -avz \
  "${REMOTE}:${REMOTE_DIR}/outputs/sam2_task_mask/${VIDEO_NAME}" \
  "${LOCAL_ROOT}/outputs/sam2_task_mask/"

echo "Done: ${LOCAL_ROOT}/outputs/sam2_task_mask/${VIDEO_NAME}"
