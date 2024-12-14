python prepare_sft_dataset.py \
    --data_input_dirs "/mnt/nvme_storage/Zehao/ColossalAI/zehao_model/database/inf-ins-3M-jsonl" \
    --tokenizer_dir "/mnt/nvme_storage/modelbase/mistral-7b-v0.1" \
    --data_output_dirs "/mnt/nvme_storage/database/Infinity-instruct-tokenized" \
    --max_length 1000000000000000019884624838656 \
    --num_spliced_dataset_bins 35 \
    --llama_version 3
