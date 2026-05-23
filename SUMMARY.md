ditto inference is in the website_real


wcss:
conda activate ditto222

preprocessing:

you ran prepare_data1.sh
inside you can prepare data and train, 
for preparing data, uncomment bash prepare_data/prepare_data.sh ${DATA_INFO_JSON} ${DATA_LIST_JSON} ${DATA_PRELOAD_PKL}
for training, the rest

get_data_info_json_for_trainset_example.py scans trainset_example/fps25_mp4 for .mp4 files, derives the stem for each, and builds parallel lists of expected artifact paths (it only creates paths, doesn't do any extraction)

Each “Processing frames: 100% …” bar is get_global_bbox iterating through every frame of a source video to detect/track the face and compute a global bounding box before cropping. There will be one bar per input clip.


Video crop: reads fps25_video_list from data_info_json, crops faces with LivePortrait, and writes cropped videos to video_list (crop_video_by_LP.py).

Audio: extracts WAV audio from the cropped videos (extract_audio_from_video.py, populates wav_list).

Features:
- HuBERT audio features from WAVs (extract_audio_feat_by_Hubert.py, fills hubert_aud_npy_list).
- Motion features from cropped videos via LivePortrait (LP) once normal, once flipped (extract_motion_feat_by_LP.py with/without --flip_flag, fills LP_pkl_list/LP_npy_list for both orientations).
- Eye features from cropped videos via MediaPipe, once normal, once flipped (extract_eye_ratio_from_video.py with/without --flip_lmk_flag, fills MP_lmk_npy_list, eye_open_npy_list, eye_ball_npy_list).
- Emotion features from cropped videos (extract_emo_feat_from_video.py, fills emo_npy_list).

Buro:
Assemble training lists: gather_data_list_json_for_train.py merges the feature paths into data_list_json with flags to include emotion and eye features and flipped variants.
Optional preload: preload_train_data_to_pkl.py reads data_list_json and serializes tensors to data_preload_pkl to speed training (with speech content, emotion, eye features, and motion dim 265).

install

pip install tyro
pip install imageio-ffmpeg==0.4.9
pip install imageio==2.33.1

było: mediapipe                 0.10.18, ma być mediapipe==0.10.8
było: protobuf                  5.29.4, a ma być protobuf==3.20.3

hsemotion==0.3.0
pip install facenet-pytorch==2.6.0
accelerate==0.33.0

DITTO_PATH=/lustre/pd01/hpc-kamada9960-1751058536/anim/ditto_train

DATA_INFO_JSON="${DITTO_PATH}/example/trainset_example/data_info.json"
DATA_LIST_JSON="${DITTO_PATH}/example/trainset_example/data_list.json"
DATA_PRELOAD_PKL="${DITTO_PATH}/example/trainset_example/data_preload.pkl"

---
zxostało zmienione dla inferenceL
cuda-python
  Attempting uninstall: numpy
    Found existing installation: numpy 1.26.4
    Uninstalling numpy-1.26.4:
      Successfully uninstalled numpy-1.26.4
  Attempting uninstall: scikit-learn
    Found existing installation: scikit-learn 1.3.2
    Uninstalling scikit-learn-1.3.2:
      Successfully uninstalled scikit-learn-1.3.2
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
facenet-pytorch 2.6.0 requires numpy<2.0.0,>=1.24.0, but you have numpy 2.2.6 which is incompatible.
accelerate 0.33.0 requires numpy<2.0.0,>=1.17, but you have numpy 2.2.6 which is incompatible.
Successfully installed colored-2.3.1 cuda-bindings-13.1.1 cuda-pathfinder-1.3.3 cuda-python-13.1.1 cython-3.2.3 numpy-2.2.6 opencv_python_headless-4.12.0.88 polygraphy-0.49.26 scikit-learn-1.8.0


conda install -y -c conda-forge mkl=2023.1.0 intel-openmp=2023.1.0


