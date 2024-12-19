python prepare_pretrain_dataset.py \
    --data_input_dirs "/data/home/scyb039/database/SlimPajama-jsonl" \
    --tokenizer_dir "/data/home/scyb039/modelbase/Mistral-7B-v0.1" \
    --data_output_dirs "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all" \
    --max_length 8192 \
    --num_spliced_dataset_bins 10 \
