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
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
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
  # --fixed_prompt='./fixed_prompt/fixed_high_quality_prompt.pt' \

## ref_pixel_values作为第一帧参考frame
## --inpaint_image_start_only控制mask设定为第一帧保留其他mask掉，但是实际没有使用

### ommited #####
  # --enable_bucket \
  # --control_ref_image="first_frame" \
  # --low_vram \






# 只有开了enable_bucket才能用enable_text_encoder_in_dataloader

# accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun/train_control.py \  # 使用 Accelerate 启动脚本，启用 bfloat16 混合精度训练
#   --config_path="config/wan2.1/wan_civitai.yaml" \                 # 模型和训练配置文件
#   --pretrained_model_name_or_path=$MODEL_NAME \                    # 预训练的 Wan2.1 模型路径
#   --train_data_dir=$DATASET_NAME \                                 # 训练数据所在的目录
#   --train_data_meta=$DATASET_META_NAME \                           # 包含每个样本元信息的 metadata.json（如路径、高宽、标签等）
#   --image_sample_size=1024 \                                       # 控制图像（如 ref image）在训练时的尺寸（方形，1024x1024）
#   --video_sample_size=256 \                                        # 视频帧将 resize 到的目标尺寸（如 256x256）
#   --token_sample_size=512 \                                        # 每帧切成 patch 后的 token grid 大小（控制 token 长度）
#   --video_sample_stride=2 \                                        # 每隔 2 帧采样一帧
#   --video_sample_n_frames=81 \                                     # 每段视频 clip 使用 81 帧（在 stride 下大概覆盖 162 帧）
#   --train_batch_size=1 \                                           # 每 GPU 的训练 batch size
#   --video_repeat=1 \                                               # 每个视频在 epoch 中重复 1 次（可用于扩增小数据集）
#   --gradient_accumulation_steps=1 \                                # 梯度累积步数（设置 >1 可模拟更大 batch）
#   --dataloader_num_workers=8 \                                     # 加载数据的线程数
#   --num_train_epochs=100 \                                         # 总共训练 100 个 epoch
#   --checkpointing_steps=50 \                                       # 每训练 50 步保存一个 checkpoint
#   --learning_rate=2e-05 \                                          # 学习率（适合微调控制模块）
#   --lr_scheduler="constant_with_warmup" \                          # 使用 warmup + constant 的学习率调度器
#   --lr_warmup_steps=100 \                                          # 前 100 步逐渐升高学习率
#   --seed=42 \                                                      # 固定随机种子，保证结果可复现
#   --output_dir="output_dir" \                                      # 模型输出的目录
#   --gradient_checkpointing \                                       # 启用梯度检查点，节省显存（训练变慢）
#   --mixed_precision="bf16" \                                       # 使用 bf16 精度训练（需硬件支持）
#   --adam_weight_decay=3e-2 \                                       # Adam 优化器的 weight decay 系数
#   --adam_epsilon=1e-10 \                                           # Adam 优化器的 epsilon，提升数值稳定性
#   --vae_mini_batch=1 \                                             # 每次给 VAE 编码器的图像数量（设置小以减少显存）
#   --max_grad_norm=0.05 \                                           # 最大梯度范数（用于裁剪防止爆炸）
#   --random_hw_adapt \                                              # 启用随机高宽比 resize 增强（训练时动态改变尺寸）
#   --training_with_video_token_length \                             # 基于 token 数量动态决定视频帧数（可增强训练灵活性）
#   --enable_bucket \                                                # 启用 resolution bucket，自动对相似尺寸归组训练
#   --uniform_sampling \                                             # 使用时间上均匀的方式从视频中采样帧
#   --low_vram \                                                     # 启用低显存优化路径（适用于小 GPU）
#   --train_mode="control_object" \                                  # 设置为控制模型训练模式（输入图像+prompt生成视频）
#   --control_ref_image="first_frame" \                              # 使用第一帧作为控制图像（可替换为其他帧或图像）
#   --trainable_modules "."                                          # 指定参与训练的模块（“.”表示全部或自动选择）