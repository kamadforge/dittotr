set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DITTO_PATH="${SCRIPT_DIR}"

INPUT_DIR="${1:-${HOME}/Dropbox/Current_research/data/HDTF/videos}"
SAVE_DIR="${2:-${DITTO_PATH}/example/trainset_hdtf}"

DATA_INFO_JSON="${SAVE_DIR}/data_info.json"
DATA_LIST_JSON="${SAVE_DIR}/data_list.json"
DATA_PRELOAD_PKL="${SAVE_DIR}/data_preload.pkl"

python example/get_data_info_json_from_video_dir.py \
    --input-dir "${INPUT_DIR}" \
    --save-dir "${SAVE_DIR}" \
    --output-json "${DATA_INFO_JSON}" \
    --validate-videos

bash prepare_data/prepare_data.sh "${DATA_INFO_JSON}" "${DATA_LIST_JSON}" "${DATA_PRELOAD_PKL}"

echo "[prepare_hdtf_data]"
echo "input_dir: ${INPUT_DIR}"
echo "save_dir: ${SAVE_DIR}"
echo "data_info_json: ${DATA_INFO_JSON}"
echo "data_list_json: ${DATA_LIST_JSON}"
echo "data_preload_pkl: ${DATA_PRELOAD_PKL}"
