# general settings
GPU=1;            # set to 01 for using gpu 0 and 1
SEED=0;           # randomness seed for sampling
CHANNELS=32;      # number of model channels
MODE='train';     # train vs sample
DATA_MODE="train" # train, test, validation data
MODEL='halfres'; #'patchddm', 'fullres', 'halfres'
DATA='GBM'      # 'HCP' or 'GBM'
FBC='W-Net_SE'     #  or 'U-Net'
# settings for sampling/inference
ITERATIONS=001;  # training iteration (as amultiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=0;# number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="MonXX_HH-MM-SS_HOSTNAME"; # tensorboard dir to be set for the evaluation
if [[ $DATA == 'HCP' ]]; then
  DATA_DIR="/data";
  IN_CHANNELS=12;
  OUT_CHANNELS=8;
  CONCAT_COORDS=False;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  HALF_RES_CROP_TRAIN=False;
elif [[ $DATA == 'GBM' ]] ; then
  DATA_DIR="/data";
  IN_CHANNELS=12;
  OUT_CHANNELS=8;
  CONCAT_COORDS=False;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  HALF_RES_CROP_TRAIN=False;
fi
#--channel_mult=1,3,4,4,4,4,4
COMMON="
--dataset=${DATA}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=1,1,2,3,4
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=1
--num_groups=${CHANNELS}
--in_channels=${IN_CHANNELS}
--concat_coords=${CONCAT_COORDS}
--out_channels=${OUT_CHANNELS}
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=True
--decoder_device_thresh=15
--FBC=${FBC}
--LR=True

"
TRAIN="
--data_dir=${DATA_DIR}
--image_size=${IMAGE_SIZE_TRAIN}
--half_res_crop=${HALF_RES_CROP_TRAIN}
--use_fp16=True
--lr=1e-5
--save_interval=50000
--num_workers=6
--devices=${GPU}
--pretrain=False
--pretrain_epochs=200
"
SAMPLE="
--data_dir=${DATA_DIR}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE_SAMPLE}
--half_res_crop=False
--use_fp16=False
--model_path=/data
--devices=${GPU}
--output_dir=/data
--num_samples=1000
--use_ddim=True
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
"
if [[ $MODE == 'train' ]]; then
  python /home/FA_CDM/FA_CDM_train.py $TRAIN $COMMON;
else
  python /home/FA_CDM/FA_CDM_sample.py $SAMPLE $COMMON;
fi
