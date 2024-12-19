python /gpfsdata/yuhong/ColossalAI/zehao_model/Colossal-LLaMA/dataset/prepare_sft_dataset.py \
    --data_input_dirs "/path/to/jsonl_files/train-*.jsonl" \
    --tokenizer_dir "/gpfsdata/yuhong/ColossalAI/zehao_model/mistral" \
    --data_output_dirs "/gpfsdata/yuhong/ColossalAI/zehao_model/database/prepared_data" \
    --max_length 4096 \
    --num_spliced_dataset_bins 10 \
    --llama_version 3
