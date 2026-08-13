#!/usr/bin/env bash
# Sync and run offline SusScore demo on remote GPU (CUDA).
# Environment variables (RTX4090-prefixed to avoid clashing with other hosts):
#   RTX4090_SSHPASS   - password for sshpass (optional)
#   RTX4090_REMOTE    - default liang@10.56.147.233
#   RTX4090_REMOTE_DIR - default ~/slerobot_sam2
#
# Example:
#   export RTX4090_SSHPASS='...'
#   ./scripts/run_susscore_demo_remote.sh
set -euo pipefail

REMOTE="${RTX4090_REMOTE:-liang@10.56.147.233}"
REMOTE_DIR="${RTX4090_REMOTE_DIR:-~/slerobot_sam2}"
if [[ -n "${RTX4090_SSHPASS:-}" ]]; then
  export SSHPASS="${RTX4090_SSHPASS}"
elif [[ -n "${SSHPASS:-}" ]]; then
  export SSHPASS
fi
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/susscore_demo_sam2}"
TASK_MASK="${TASK_MASK:-outputs/sam2_task_mask/M_task_side_ep20_trigger.npy}"
CLEAN_EPISODE="${CLEAN_EPISODE:-4}"
TRIGGER_EPISODE="${TRIGGER_EPISODE:-20}"
FULL_EPISODE="${FULL_EPISODE:-1}"
DEVICE="${DEVICE:-cuda}"

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

if [[ ! -f "${LOCAL_ROOT}/${TASK_MASK}" ]]; then
  echo "Missing task mask: ${LOCAL_ROOT}/${TASK_MASK}" >&2
  exit 1
fi

echo "==> Prepare remote dir ${REMOTE}:${REMOTE_DIR}"
run_ssh "mkdir -p ${REMOTE_DIR}/src ${REMOTE_DIR}/outputs/sam2_task_mask ${REMOTE_DIR}/${OUTPUT_DIR}"

echo "==> Sync code + task mask"
run_rsync -avz \
  --exclude '__pycache__' \
  "${LOCAL_ROOT}/src/" "${REMOTE}:${REMOTE_DIR}/src/"
run_rsync -avz \
  "${LOCAL_ROOT}/pyproject.toml" \
  "${REMOTE}:${REMOTE_DIR}/"
run_rsync -avz \
  "${LOCAL_ROOT}/${TASK_MASK}" \
  "${REMOTE}:${REMOTE_DIR}/${TASK_MASK}"

FULL_FLAG=""
if [[ "${FULL_EPISODE}" == "1" ]]; then
  FULL_FLAG="--full_episode"
fi

echo "==> Run SusScore demo on remote GPU (clean ep ${CLEAN_EPISODE}, trigger ep ${TRIGGER_EPISODE})"
run_ssh bash -s <<EOF
set -euo pipefail
cd ${REMOTE_DIR}

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip wheel

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  opencv-python-headless \
  einops \
  draccus \
  pyyaml \
  numpy \
  pandas \
  pyarrow \
  pillow \
  datasets \
  huggingface_hub \
  packaging \
  imageio \
  imageio-ffmpeg \
  av \
  safetensors \
  packaging

python -m pip install -e .

export PYTHONPATH="\${PWD}/src:\${PYTHONPATH:-}"
export HF_HOME="\${PWD}/.cache/huggingface"

KMP_DUPLICATE_LIB_OK=TRUE python src/slerobot/scripts/slerobot_offline_susscore_demo.py \
  --output_dir ${OUTPUT_DIR} \
  --task_mask_path ${TASK_MASK} \
  --clean_episode ${CLEAN_EPISODE} \
  --trigger_episode ${TRIGGER_EPISODE} \
  --device ${DEVICE} \
  ${FULL_FLAG}
EOF

echo "==> Fetch videos"
mkdir -p "${LOCAL_ROOT}/${OUTPUT_DIR}"
run_rsync -avz \
  "${REMOTE}:${REMOTE_DIR}/${OUTPUT_DIR}/" \
  "${LOCAL_ROOT}/${OUTPUT_DIR}/"

echo "Done. Local outputs:"
ls -lh "${LOCAL_ROOT}/${OUTPUT_DIR}/"*.mp4 2>/dev/null || true
