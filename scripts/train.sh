MODEL_NAME="./models/Qwen2.5-0.5B"
# MODEL_NAME="Qwen/Qwen2.5-0.5B"
# MODEL_NAME="mistralai/Mistral-7B-v0.1"

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
accelerate launch \
  --config_file accelerate_config.yaml \
  -m embedding.main \
  --model_name_or_path=$MODEL_NAME \
  --per_device_train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_new_tokens=1 \
  --max_steps=5 \
  --logit_temp=1 \
  --wandb_project="debug" \
  --model_type="generation" \
  --pooling_method="generate_mean" \
  --save_steps=5 \
  --data_sampling_rate=0.01 \
  --reg_weight=1 \
  --output_dir ckpts/debug-QWEN0.5B
