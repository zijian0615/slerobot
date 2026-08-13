#!/usr/bin/env bash
# Run SusScore demo + cyclic defense on local eval dataset via remote GPU.
# Dataset: input/eval_xa3_clean123_bad2_work (single episode 0)
set -euo pipefail

REMOTE="${RTX4090_REMOTE:-liang@10.56.147.233}"
REMOTE_DIR="${RTX4090_REMOTE_DIR:-~/slerobot_sam2}"
if [[ -n "${RTX4090_SSHPASS:-}" ]]; then
  export SSHPASS="${RTX4090_SSHPASS}"
elif [[ -n "${SSHPASS:-}" ]]; then
  export SSHPASS
fi
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

LOCAL_DATASET="${LOCAL_DATASET:-input/eval_xa3_clean123_bad2_work}"
REMOTE_DATASET="${REMOTE_DIR}/${LOCAL_DATASET}"
TASK_MASK="${TASK_MASK:-outputs/sam2_task_mask/M_task_side_ep20_trigger.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/susscore_demo_eval_xa3}"
CYCLIC_DIR="${CYCLIC_DIR:-outputs/susscore_cyclic_defense_eval_xa3}"
DEVICE="${DEVICE:-cuda}"
POLICY_PATH="${POLICY_PATH:-zijian2022/xa3_clean123_bad2_work}"
CALIB_REPO="${CALIB_REPO:-zijian2022/xa3_clean123_bad2}"

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

if [[ ! -d "${LOCAL_ROOT}/${LOCAL_DATASET}" ]]; then
  echo "Missing dataset: ${LOCAL_ROOT}/${LOCAL_DATASET}" >&2
  exit 1
fi
if [[ ! -f "${LOCAL_ROOT}/${TASK_MASK}" ]]; then
  echo "Missing task mask: ${LOCAL_ROOT}/${TASK_MASK}" >&2
  exit 1
fi

echo "==> Prepare remote dirs"
run_ssh "mkdir -p ${REMOTE_DIR}/src ${REMOTE_DIR}/outputs/sam2_task_mask ${REMOTE_DIR}/${OUTPUT_DIR} ${REMOTE_DIR}/${CYCLIC_DIR} ${REMOTE_DIR}/input"

echo "==> Sync code + task mask + eval dataset"
run_rsync -avz --exclude '__pycache__' "${LOCAL_ROOT}/src/" "${REMOTE}:${REMOTE_DIR}/src/"
run_rsync -avz "${LOCAL_ROOT}/pyproject.toml" "${REMOTE}:${REMOTE_DIR}/"
run_rsync -avz "${LOCAL_ROOT}/${TASK_MASK}" "${REMOTE}:${REMOTE_DIR}/${TASK_MASK}"
run_rsync -avz \
  --exclude '.git' \
  --exclude '.cache' \
  "${LOCAL_ROOT}/${LOCAL_DATASET}/" \
  "${REMOTE}:${REMOTE_DATASET}/"

echo "==> Run SusScore demo + cyclic defense on remote GPU"
run_ssh bash -s <<EOF
set -euo pipefail
cd ${REMOTE_DIR}
source .venv/bin/activate
export PYTHONPATH="\${PWD}/src:\${PYTHONPATH:-}"
export HF_HOME="\${PWD}/.cache/huggingface"

KMP_DUPLICATE_LIB_OK=TRUE python src/slerobot/scripts/slerobot_offline_susscore_demo.py \
  --dataset.repo_id eval_xa3_clean123_bad2_work \
  --dataset.root ${LOCAL_DATASET} \
  --calib_repo_id ${CALIB_REPO} \
  --policy.path ${POLICY_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --task_mask_path ${TASK_MASK} \
  --trigger_episode 0 \
  --run_label eval \
  --full_episode \
  --device ${DEVICE}

KMP_DUPLICATE_LIB_OK=TRUE python src/slerobot/scripts/slerobot_offline_cyclic_defense.py \
  --dataset.repo_id eval_xa3_clean123_bad2_work \
  --dataset.root ${LOCAL_DATASET} \
  --calib_repo_id ${CALIB_REPO} \
  --policy.path ${POLICY_PATH} \
  --output_dir ${CYCLIC_DIR} \
  --task_mask_path ${TASK_MASK} \
  --episode 0 \
  --device ${DEVICE}
EOF

echo "==> Fetch results"
mkdir -p "${LOCAL_ROOT}/${OUTPUT_DIR}" "${LOCAL_ROOT}/${CYCLIC_DIR}"
run_rsync -avz "${REMOTE}:${REMOTE_DIR}/${OUTPUT_DIR}/" "${LOCAL_ROOT}/${OUTPUT_DIR}/"
run_rsync -avz "${REMOTE}:${REMOTE_DIR}/${CYCLIC_DIR}/" "${LOCAL_ROOT}/${CYCLIC_DIR}/"

echo "Done."
ls -lh "${LOCAL_ROOT}/${OUTPUT_DIR}/"*.mp4 2>/dev/null || true
ls -lh "${LOCAL_ROOT}/${CYCLIC_DIR}/"* 2>/dev/null || true
