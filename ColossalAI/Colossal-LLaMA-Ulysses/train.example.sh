#!/bin/bash
set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv |
        tail -n +2 |
        nl -v 0 |
        tee /dev/tty |
        sort -g -k 2 |
        awk '{print $1}' |
        head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 8

# PROJECT_NAME="mistral_pretrain"
# PARENT_SAVE_DIR="/gpfsdata/yuhong/ColossalAI/zehao_model/mistral_modelsave"
# PARENT_TENSORBOARD_DIR="/gpfsdata/yuhong/ColossalAI/zehao_model/mistral_tensorsave"
# PARENT_CONFIG_FILE=""
# PRETRAINED_MODEL_PATH=""

declare -a dataset=(
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized"
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00000-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00001-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00002-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00003-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00004-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00005-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00006-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00007-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00008-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00009-of-00015.arrow",
    # "/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized/train/data-00010-of-00015.arrow",
)

# TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
# FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
# SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
# TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
# CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"
# --hostfile hostfile --master_port 30013
colossalai run --nproc_per_node 8  train.py \
    --pretrained /gpfsdata/yuhong/ColossalAI/zehao_model/mistral \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 400 \
    --save_dir "/gpfsdata/yuhong/ColossalAI/zehao_model/save/model" \
    --tensorboard_dir "/gpfsdata/yuhong/ColossalAI/zehao_model/save/tensorboard" \
    --config_file "/gpfsdata/yuhong/ColossalAI/zehao_model/mistral/config.json" \
    --num_epochs 1 \
    --batch_size 4 \
    --lr 1e-4 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_grad_checkpoint \
    --use_flash_attn \
    --pad_token "unk"

