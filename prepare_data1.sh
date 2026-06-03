set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DITTO_PATH="${SCRIPT_DIR}"

MODE="${1:-all}"
case "${MODE}" in
    all|prepare|train)
        ;;
    -h|--help|help)
        echo "Usage: bash prepare_data1.sh [all|prepare|train]"
        echo "  all      prepare data, then train (default)"
        echo "  prepare  prepare data only"
        echo "  train    train only, using existing data_list.json and data_preload.pkl"
        exit 0
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Usage: bash prepare_data1.sh [all|prepare|train]" >&2
        exit 1
        ;;
esac

DATA_INFO_JSON="${DITTO_PATH}/example/trainset_example/data_info.json"
DATA_LIST_JSON="${DITTO_PATH}/example/trainset_example/data_list.json"
DATA_PRELOAD_PKL="${DITTO_PATH}/example/trainset_example/data_preload.pkl"

if [[ "${MODE}" == "all" || "${MODE}" == "prepare" ]]; then

## Prepare data_info.json
python example/get_data_info_json_for_trainset_example.py
# you will get `example/trainset_example/data_info.json`


## Process Videos into Training Features
bash prepare_data/prepare_data.sh ${DATA_INFO_JSON} ${DATA_LIST_JSON} ${DATA_PRELOAD_PKL}

python - <<PY
import json, os, sys
path = "${DATA_LIST_JSON}"
if not os.path.isfile(path):
    sys.exit(f"data_list_json not found: {path}")
with open(path) as f:
    data = json.load(f)
if not data:
    sys.exit(f"data_list_json is empty: {path}. Check prepare_data logs.")
print(f"Prepared {len(data)} sequences.")
PY

fi

if [[ "${MODE}" == "prepare" ]]; then
    exit 0
fi

python - <<PY
import os, pickle, sys
data_list_json = "${DATA_LIST_JSON}"
data_preload_pkl = "${DATA_PRELOAD_PKL}"
if not os.path.isfile(data_list_json):
    sys.exit(f"data_list_json not found: {data_list_json}. Run: bash prepare_data1.sh prepare")
if not os.path.isfile(data_preload_pkl):
    sys.exit(f"data_preload_pkl not found: {data_preload_pkl}. Run: bash prepare_data1.sh prepare")
v_list, idx_map = pickle.load(open(data_preload_pkl, "rb"))
if not v_list or not idx_map:
    sys.exit(f"data_preload_pkl empty: {data_preload_pkl}. Run: bash prepare_data1.sh prepare")
print(f"Using prepared preload: {len(v_list)} videos, {len(idx_map)} sequences.")
PY

## MotionDiT Training
cd MotionDiT

EXP_DIR="${DITTO_PATH}/example/exp_dir"
EXP_NAME="exp_trainset_example"

accelerate launch train.py \
    --experiment_dir ${EXP_DIR} \
    --experiment_name ${EXP_NAME} \
    --use_sc \
    --use_last_frame \
    --use_last_frame_loss \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --audio_feat_dim 1103 \
    --motion_feat_dim 265 \
    --batch_size 100 \
    --num_workers 8 \
    --epochs 100 \
    --save_ckpt_freq 1 \
    --data_list_json ${DATA_LIST_JSON} \
    --data_preload \
    --data_preload_pkl ${DATA_PRELOAD_PKL}

# training outputs in `example/exp_dir/exp_trainset_example`
