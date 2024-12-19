#!/bin/bash

# NCCL IB environment variables
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
    export FIRST_N_GPU_IDS="0,1,2,3,4,5,6,7"
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    # export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 8

declare -a dataset=(
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00000"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00001"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00002"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00003"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00004"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00005"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00006"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00007"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00008"
    "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow/part-00009"
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 8  --master_port=29501 train.py \
    --pretrained /mnt/nvme_storage/modelbase/gla-chat\
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 50 \
    --save_dir "/mnt/nvme_storage/modelbase/saved_gla-ins" \
    --tensorboard_dir "/mnt/nvme_storage/modelbase/saved_ten" \
    --config_file "/mnt/nvme_storage/modelbase/gla-chat/run_config.json" \
    --num_epochs 1 \
    --accumulation_steps 16 \
    --batch_size 1 \
    --lr 5e-5 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --use_grad_checkpoint \
    --use_neft \
    --pad_token "eos"\
    --use_grad_checkpoint \
    --benchmark
