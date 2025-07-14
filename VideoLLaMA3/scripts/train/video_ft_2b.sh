#!/bin/bash
# Environment Variables
# 单机训练参数
export HF_ENDPOINT=https://hf-mirror.com
WORLD_SIZE=1
NPROC_PER_NODE=4
MASTER_ADDR="127.0.0.1"
MASTER_PORT=16667
RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=32   # 128
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

# Log Arguments
export WANDB_PROJECT=videollama3_2b_local_new_2stage
PRECEDING_RUN_NAME=download_model
RUN_NAME=ft_stageI_batch32_180f_2e
DATA_DIR=/home/zhouting/SurveillanceVideoFireAnalysis/annotation_data/videollama_format
OUTP_DIR=/data/zhouting/outputs/fire_analysis/models

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama3/train.py \
    --deepspeed scripts/zero1.json \
    --lora_enable False \
    --model_type videollama3_qwen2 \
    --model_path /data/zhouting/models/videollama3_2b_local \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path /home/zhouting/SurveillanceVideoFireAnalysis/annotation_data/training_data/StageI_training_data_6_24_cutlongvideo20min_fix.json \
    --data_folder /data/zhouting/FireVAD/01_videos_transcoded/STAGE_I_VIDEO_CLIPS \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 180 \
    --model_max_length 16384 \
    --mm_max_length 10240 \
    --use_token_compression True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --num_train_epochs 2 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --llm_lr 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    --dataset_cache_dir /data/zhouting/Dataset_Cache/.cache \
    --dataloader_persistent_workers False