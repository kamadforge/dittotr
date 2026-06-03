ditto inference is in the website_real

# QUICK

**Train**

cd /home/kamil/Dropbox/Current_research/t3D/ditto_train

1. Prepare HDTF data
bash prepare_hdtf_data.sh

2. Train
bash train_hdtf.sh example/trainset_hdtf 100 64


Checkpoints:
example/exp_dir/exp_hdtf/ckpts/

Example:
train_100.pt


**Test / Inference**

cd /home/kamil/Dropbox/Current_research/t3D/ditto-talkinghead

1. Create inference cfg for checkpoint
python scripts/create_train_cfg.py \
  --checkpoint /home/kamil/Dropbox/Current_research/t3D/ditto_train/example/exp_dir/exp_hdtf/ckpts/train_100.pt \
  --output generated_cfg/hdtf_train_100_cfg.pkl

2. Run inference
python inference.py \
  --data_root ./checkpoints/ditto_pytorch \
  --cfg_pkl ./generated_cfg/hdtf_train_100_cfg.pkl \
  --audio_path ./example/audio.wav \
  --source_path ./example/image.png \
  --output_path ./generated_cfg/hdtf_train_100.mp4

Output:

/home/kamil/Dropbox/Current_research/t3D/ditto-talkinghead/generated_cfg/hdtf_train_100.mp4

# ENVS


hyperbook:
py3t
bash prepare_data1.sh

wcss:
conda activate ditto222

preprocessing:

you ran prepare_data1.sh
inside you can prepare data and train, 
for preparing data, uncomment bash prepare_data/prepare_data.sh ${DATA_INFO_JSON} ${DATA_LIST_JSON} ${DATA_PRELOAD_PKL}
for training, the rest



prepare_data1.sh uses the bundled example dataset in:

example/trainset_example/fps25_mp4/
There are 5 source videos:

-1wCGAxqT_4+000000_000177+p0.mp4
Clip+WSGW-7gHS6A+P0+C1+F2551-2669.mp4
Clip+XCHmk9jOypY+P0+C0+F6905-7009.mp4
Clip+eynXzhySluI+P0+C0+F20600-20730.mp4
Clip+uWZRSkqJ--g+P0+C2+F8754-8865.mp4
The script converts those into generated training features under:

example/trainset_example/video/
example/trainset_example/wav/
example/trainset_example/hubert_aud_npy/
example/trainset_example/LP_npy/
example/trainset_example/LP_npy_flip/
example/trainset_example/emo_npy/
example/trainset_example/eye_open_npy/
example/trainset_example/eye_open_npy_flip/
example/trainset_example/eye_ball_npy/
example/trainset_example/eye_ball_npy_flip/
Then data_list.json contains 10 entries: the 5 original clips plus 5 flipped versions. Training loads:

example/trainset_example/data_preload.pkl
That preload currently contains:

num_v: 10
num_seq: 778
So this is not using an external dataset. It is training only on the tiny example set shipped in example/trainset_example.

output logs

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


# Training:

This pipeline does data preparation + short training run for a Ditto/LivePortrait-style talking-head / motion model.

Main stages:

Creates dataset metadata
saved: .../data_info.json

It scans the example training data and writes metadata.

Extracts face / landmark / motion features
It loads LivePortrait-related modules:
appearance_extractor.pth
motion_extractor.pth
warp_network.pth
decoder.pth
stitch_network.pth

These are used to extract appearance, motion, warping, decoding, and stitching features from video frames.

Runs face analysis and landmarks
You see:
FaceAnalysisDIY warmup
LandmarkRunner warmup
Processing frames...

So it detects faces and landmarks frame-by-frame. Some of this falls back to CPU because ONNX CUDA cannot load libcudnn.so.9.

Extracts extra conditioning features
It also uses:
use_emo: True
use_eye_open: True
use_eye_ball: True
use_sc: True

So the training data includes emotion features, eye openness, eyeball/gaze features, and probably scale/crop or shape coefficients.

Builds a preload cache
data_preload.pkl
load [num_v: 10, num_seq: 778]

It converts the processed dataset into a cached .pkl file so training can load it quickly. It prepared 10 videos/sequences sources and 778 training sequences.

Starts training
Then it launches training with accelerate:
epochs: 3
batch_size: 100
seq_frames: 80
audio_feat_dim: 1103
motion_feat_dim: 265

So the model learns to map audio/features to motion features over 80-frame sequences.

Initializes the model
Model has 47788297 parameters

So it trains a ~47.8M parameter model, probably the LMDM motion diffusion/motion prediction model.

Runs 3 epochs and saves checkpoints
[MODEL SAVED at Epoch 1]
[MODEL SAVED at Epoch 2]
[MODEL SAVED at Epoch 3]

So the run completed successfully.


The training checkpoints are here:

example/exp_dir/exp_trainset_example/ckpts/

The latest checkpoint from the 3-epoch example run is:

example/exp_dir/exp_trainset_example/ckpts/train_3.pt

Other output files:

example/exp_dir/exp_trainset_example/loss.log
example/exp_dir/exp_trainset_example/opt.pkl

# inference for training

folder in t3D/ditto-talkinghead

python inference.py \
  --data_root /home/kamil/Dropbox/Current_research/t3D/ditto-talkinghead/checkpoints/ditto_pytorch \
  --cfg_pkl /home/kamil/Dropbox/Current_research/t3D/ditto_train/example/exp_dir/exp_trainset_example/v0.4_hubert_cfg_train3.pkl \
  --audio_path /home/kamil/Dropbox/Current_research/t3D/ditto-talkinghead/example/audio.wav \
  --source_path /home/kamil/Dropbox/Current_research/t3D/ditto-talkinghead/example/image.png \
  --output_path /home/kamil/Dropbox/Current_research/t3D/ditto_train/example/exp_dir/exp_trainset_example/infer_train3.mp4

  /home/kamil/Dropbox/Current_research/t3D/ditto_train/example/exp_dir/exp_trainset_example/infer_train3.mp4

to modify

python inference.py \
  --data_root /home/kamil/Dropbox/Current_research/t3D/ditto-talkinghead/checkpoints/ditto_pytorch \
  --cfg_pkl /home/kamil/Dropbox/Current_research/t3D/ditto_train/example/exp_dir/exp_trainset_example/v0.4_hubert_cfg_train3.pkl \
  --audio_path /home/kamil/Dropbox/Current_research/t3D/ditto-talkinghead/example/audio.wav \
  --source_path /home/kamil/Dropbox/Current_research/t3D/ditto-talkinghead/example/image.png \
  --output_path /home/kamil/Dropbox/Current_research/t3D/ditto_train/example/exp_dir/exp_trainset_example/infer_train3.mp4




  ## create cfg
  in ditto_talkinghead/scripts

python scripts/create_train_cfg.py \
  --checkpoint /home/kamil/Dropbox/Current_research/t3D/ditto_train/example/exp_dir/exp_trainset_example/ckpts/train_100.pt


# data


Bare minimum for visible improvement:
50-100 clean videos, 1-3 hours total, 200-500 epochs

Reasonable single-speaker/person-specific model:
5-20 hours of clean talking-head video, 300-800 epochs

Closer to original general checkpoint quality:
hundreds to thousands of hours or at least many hundreds/thousands of diverse clips, ~500+ epochs




basic setup

5 source videos
10 entries after flip augmentation
778 training windows
3 epochs
batch_size 100


From each source .mp4, prepare_hdtf_data.sh creates these training artifacts:

video/*.mp4
Cropped 512x512 face video.

wav/*.wav
Extracted 16 kHz audio from the cropped video.

hubert_aud_npy/*.npy
HuBERT audio features used as speech conditioning.

LP_pkl/*.pkl
LP_npy/*.npy
LivePortrait motion features from the cropped video.

LP_pkl_flip/*.pkl
LP_npy_flip/*.npy
Motion features for horizontally flipped version.

MP_lmk_npy/*.npy
MP_lmk_npy_flip/*.npy
MediaPipe facial landmarks, normal and flipped.

eye_open_npy/*.npy
eye_open_npy_flip/*.npy
Eye openness features.

eye_ball_npy/*.npy
eye_ball_npy_flip/*.npy
Eye-ball / gaze features.

emo_npy/*.npy
Emotion features.

Then it creates dataset index/cache files:

data_info.json
Mapping from source videos to all expected output feature paths.

data_list.json
Final list of usable training samples after checking required features exist.

data_preload.pkl
Preloaded training cache used by MotionDiT training.

For HDTF, all of this is under:

example/trainset_hdtf/