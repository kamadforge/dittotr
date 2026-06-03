set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DITTO_PATH="${SCRIPT_DIR}"

DATASET_DIR="${1:-${DITTO_PATH}/example/trainset_hdtf}"
EPOCHS="${2:-500}"
BATCH_SIZE="${3:-100}"

DATA_LIST_JSON="${DATASET_DIR}/data_list.json"
DATA_PRELOAD_PKL="${DATASET_DIR}/data_preload.pkl"

python - <<PY
import os, pickle, sys
data_list_json = "${DATA_LIST_JSON}"
data_preload_pkl = "${DATA_PRELOAD_PKL}"
if not os.path.isfile(data_list_json):
    sys.exit(f"data_list_json not found: {data_list_json}. Run: bash prepare_hdtf_data.sh")
if not os.path.isfile(data_preload_pkl):
    sys.exit(f"data_preload_pkl not found: {data_preload_pkl}. Run: bash prepare_hdtf_data.sh")
v_list, idx_map = pickle.load(open(data_preload_pkl, "rb"))
if not v_list or not idx_map:
    sys.exit(f"data_preload_pkl empty: {data_preload_pkl}. Run: bash prepare_hdtf_data.sh")
print(f"Using HDTF preload: {len(v_list)} videos, {len(idx_map)} sequences.")
PY

cd "${DITTO_PATH}/MotionDiT"

EXP_DIR="${DITTO_PATH}/example/exp_dir"
EXP_NAME="exp_hdtf"

accelerate launch train.py \
    --experiment_dir "${EXP_DIR}" \
    --experiment_name "${EXP_NAME}" \
    --use_sc \
    --use_last_frame \
    --use_last_frame_loss \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --audio_feat_dim 1103 \
    --motion_feat_dim 265 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 8 \
    --epochs "${EPOCHS}" \
    --save_ckpt_freq 10 \
    --data_list_json "${DATA_LIST_JSON}" \
    --data_preload \
    --data_preload_pkl "${DATA_PRELOAD_PKL}"

echo "[train_hdtf]"
echo "dataset_dir: ${DATASET_DIR}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "output_dir: ${EXP_DIR}/${EXP_NAME}"
