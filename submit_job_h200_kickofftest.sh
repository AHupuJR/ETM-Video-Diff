#!/bin/bash
#SBATCH --job-name=kickofftest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=10000 
#SBATCH --output=/home/lei_sun/projects/evdiff/ETM-Video-Diff/slurm_out/%x_%j.out
#SBATCH --error=/home/lei_sun/projects/evdiff/ETM-Video-Diff/slurm_out/%x_%j.err
#SBATCH --time=96:00:00         # Job timeout

eval "$(micromamba shell hook --shell bash)"
micromamba activate evdiff
cd /home/lei_sun/projects/evdiff/ETM-Video-Diff


export MODEL_NAME="/work/lei_sun/models/Wan2.1-Fun-1.3B-Control"
export DATASET_NAME="./datasets/toy_dataset_control/" # TODO
export DATASET_META_NAME="./datasets/toy_dataset_control/json_of_toy_dataset_control.json" # TODO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun/train_control.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=4 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=10000 \
  --checkpointing_steps=1000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --uniform_sampling \
  --train_mode="control_object" \
  --trainable_modules "."\
  --enable_inpaint \
  --inpaint_image_start_only \
  --fixed_prompt='./fixed_prompt/fixed_high_quality_prompt.pt' \
