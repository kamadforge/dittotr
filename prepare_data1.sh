set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DITTO_PATH="${SCRIPT_DIR}"


## Prepare data_info.json
python example/get_data_info_json_for_trainset_example.py
# you will get `example/trainset_example/data_info.json`


## Process Videos into Training Features
DATA_INFO_JSON="${DITTO_PATH}/example/trainset_example/data_info.json"
DATA_LIST_JSON="${DITTO_PATH}/example/trainset_example/data_list.json"
DATA_PRELOAD_PKL="${DITTO_PATH}/example/trainset_example/data_preload.pkl"

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
    --epochs 3 \
    --save_ckpt_freq 1 \
    --data_list_json ${DATA_LIST_JSON} \
    --data_preload \
    --data_preload_pkl ${DATA_PRELOAD_PKL}

# training outputs in `example/exp_dir/exp_trainset_example`
