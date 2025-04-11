# Wan2.1-Fun 控制训练文档（中文说明）

## 🧠 训练代码说明

Wan-Fun 支持多种训练方式，可以选择是否使用 **DeepSpeed** 来大幅节省显存资源。

---

## 🗂 数据格式说明

在 Wan-Fun 中，`metadata_control.json` 格式如下，与普通格式不同之处在于增加了 `control_file_path` 字段。

建议使用 [DWPose](https://github.com/IDEA-Research/DWPose) 作为关键点检测工具来生成控制文件。

```json
[
  {
    "file_path": "train/00000001.mp4",
    "control_file_path": "control/00000001.mp4",
    "text": "一群穿着西装和太阳镜的年轻男子走在城市街道上。",
    "type": "video"
  },
  {
    "file_path": "train/00000002.jpg",
    "control_file_path": "control/00000002.jpg",
    "text": "一群穿着西装和太阳镜的年轻男子走在城市街道上。",
    "type": "image"
  }
]
```

---

## ⚙️ 重要参数解释

| 参数名 | 说明 |
|--------|------|
| `enable_bucket` | 开启 Bucket 分辨率分组训练，避免中心裁剪 |
| `random_frame_crop` | 视频帧随机裁剪，模拟不同帧率 |
| `random_hw_adapt` | 图像与视频自动高宽缩放 |
| `training_with_video_token_length` | 根据 token 长度训练，可控图像/视频空间分辨率与帧数 |

### 🧮 Token 长度示例：

- `video_sample_n_frames = 49`
- `token_sample_size = 512`

在不同分辨率下视频帧数估算如下：

| 分辨率 | 帧数（近似） |
|--------|--------------|
| 512×512 | 49 帧 |
| 768×768 | 21 帧 |
| 1024×1024 | 9 帧 |

### 参数具体解释
一些 `.sh` 文件中的参数可能会让人困惑，下面是对这些参数的解释：

- `enable_bucket`：用于启用分桶训练（bucket training）。启用后，模型不会对图像和视频进行中心裁剪，而是根据分辨率对图像和视频进行分桶，并对整个图像或视频进行训练。
- `random_frame_crop`：用于对视频帧进行随机裁剪，以模拟不同帧数的视频。
- `random_hw_adapt`：用于启用图像和视频的自动高宽缩放。当启用 `random_hw_adapt` 时，训练图像的高和宽将在 `min(video_sample_size, 512)` 到 `image_sample_size` 之间变化。训练视频的高和宽也在同一范围内变化。
  - 例如，当启用 `random_hw_adapt`，设置 `video_sample_n_frames=49`，`video_sample_size=1024`，`image_sample_size=1024`，则训练图像的分辨率为 `512x512` 到 `1024x1024`，训练视频的分辨率为 `512x512x49` 到 `1024x1024x49`。
  - 又如，若设置 `video_sample_n_frames=49`，`video_sample_size=1024`，`image_sample_size=256`，则图像分辨率为 `256x256` 到 `1024x1024`，视频分辨率为 `256x256x49`。
- `training_with_video_token_length`：表示根据 token 长度进行训练。图像和视频的高宽范围将在 `video_sample_size` 和 `image_sample_size` 之间变化。
  - 例如，启用该选项时，若设置 `video_sample_n_frames=49`，`token_sample_size=1024`，`video_sample_size=1024`，`image_sample_size=256`，则图像分辨率为 `256x256` 到 `1024x1024`，视频分辨率为 `256x256x49` 到 `1024x1024x49`。
  - 又如，若设置 `video_sample_n_frames=49`，`token_sample_size=512`，`video_sample_size=1024`，`image_sample_size=256`，则视频分辨率为 `256x256x49` 到 `1024x1024x9`。
  - 512x512 分辨率、49 帧的视频的 token 长度约为 13,312，因此应设置 `token_sample_size=512`。
    - 在 512x512 分辨率下，帧数为 49（约等于 `512 * 512 * 49 / 512 / 512`）。
    - 在 768x768 分辨率下，帧数为 21（约等于 `512 * 512 * 49 / 768 / 768`）。
    - 在 1024x1024 分辨率下，帧数为 9（约等于 `512 * 512 * 49 / 1024 / 1024`）。
    - 利用这些分辨率与对应帧数的组合，模型可以生成不同尺寸的视频。
- `resume_from_checkpoint`：用于从上一次的训练断点继续训练。可以填写路径，也可以填写 `"latest"` 来自动选择最后一个可用的 checkpoint。
---

## 🚀 训练命令（无 DeepSpeed）

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-14B-Control"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun/train_control.py   --config_path="config/wan2.1/wan_civitai.yaml"   --pretrained_model_name_or_path=$MODEL_NAME   --train_data_dir=$DATASET_NAME   --train_data_meta=$DATASET_META_NAME   --image_sample_size=1024   --video_sample_size=256   --token_sample_size=512   --video_sample_stride=2   --video_sample_n_frames=81   --train_batch_size=1   --video_repeat=1   --gradient_accumulation_steps=1   --dataloader_num_workers=8   --num_train_epochs=100   --checkpointing_steps=50   --learning_rate=2e-05   --lr_scheduler="constant_with_warmup"   --lr_warmup_steps=100   --seed=42   --output_dir="output_dir"   --gradient_checkpointing   --mixed_precision="bf16"   --adam_weight_decay=3e-2   --adam_epsilon=1e-10   --vae_mini_batch=1   --max_grad_norm=0.05   --random_hw_adapt   --training_with_video_token_length   --enable_bucket   --uniform_sampling   --low_vram   --train_mode="control_object"   --control_ref_image="first_frame"   --trainable_modules "."
```

---

## 💾 使用 DeepSpeed 加速训练

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-14B-Control"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_fun/train_control.py   ... # 其他参数与上方相同
```

---

## 🧊 DeepSpeed Zero-3（大模型推荐）

适用于训练超大模型（如 14B）：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-14B-Control"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed   --deepspeed_config_file config/zero_stage2.1_config.json --deepspeed_multinode_launcherr standard scripts/wan2.1_fun/train_control.py   --config_path="config/wan2.1_fun/wan_civitai.yaml"   ...
  --train_mode="inpaint"   --low_vram
```

完成训练后，将模型转换为标准 bf16 格式：

```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{your-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

---


