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
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00000"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00001"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00002"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00003"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00004"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00005"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00006"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00007"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00008"
    "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00009"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00000"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00001"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00002"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00003"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00004"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00005"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00006"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00007"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00008"
    # "/data/home/scyb039/database/SlimPj_4_CoAI/arrow/part-00009"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00000"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00001"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00002"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00003"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00004"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00005"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00006"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00007"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00008"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI/arrow/part-00009"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00000"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00001"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00002"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00003"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00004"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00005"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00006"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00007"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00008"
    # "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_small/arrow/part-00009"
)

# TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
# FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
# SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
# TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
# CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"
# --hostfile hostfile --master_port 30013
colossalai run --nproc_per_node 8  train.py \
    --pretrained /data/home/scyb039/modelbase/Mistral-7B-v0.1 \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 400 \
    --save_dir "/data/home/scyb039/modelbase/saved_ckpt/lora-boss-gla" \
    --tensorboard_dir "/data/home/scyb039/modelbase/saved_ckpt/lora-boss-gla/tensorboard" \
    --config_file "/data/home/scyb039/modelbase/saved_ckpt/lora-boss-gla/run_config.json" \
    --num_epochs 1 \
    --accumulation_steps 16 \
    --batch_size 2 \
    --lr 3e-5 \
    --max_length 8192 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.1 \
    --warmup_steps 1500 \
    --use_grad_checkpoint \
    --pad_token "unk" \
    --device_num 8\
    --steps_per_epoch 10000
    # --benchmark
