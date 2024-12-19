python prepare_sft_dataset.py \
    --data_input_dirs "/mnt/nvme_storage/Zehao/ColossalAI/zehao_model/database/inf-ins-3M-jsonl" \
    --tokenizer_dir "/mnt/nvme_storage/modelbase/gla-chat" \
    --data_output_dirs "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla" \
    --max_length 2048 \
    --num_spliced_dataset_bins 10 \
    --llama_version 4
