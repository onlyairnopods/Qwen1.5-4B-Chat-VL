# --nnodes 1 --nproc_per_node 4 --master_port 25641

# 使用这三个参数可以异步的处理数据：
# --dataloader_pin_memory True \
# --dataloader_num_workers 10 \
# --dataloader_persistent_workers True \

SAVE_DIR="../saved_output_user_lora"
mkdir -p $SAVE_DIR

deepspeed --include localhost:0,5 train.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path ../Qwen1.5-4B-Chat-clip-vit-large-patch14-336/model \
    --train_type use_lora \
    --data_path "/home/yjp/.cache/huggingface/hub/datasets--liuhaotian--LLaVA-CC3M-Pretrain-595K/snapshots/814894e93db9e12a1dee78b9669e20e8606fd590" \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --output_dir $SAVE_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers True \
    2>&1 | tee $SAVE_DIR/train.log

# --model_max_length 2048

# --save_strategy "steps" \
# --save_steps 10 \
# --save_steps 1000 \
# --save_strategy "epoch" \