mkdir /data/home/scyb039/database/Splimpajama-DE/logs
mkdir /data/home/scyb039/database/Splimpajama-DE/data
mkdir /data/home/scyb039/database/Splimpajama-DE/data/slimpajama
mkdir /data/home/scyb039/database/Splimpajama-DE/data/slimpajama/per_source_downsample
# cd data_engineering

PATH_TO_SLIMPAJAMA=/data/home/scyb039/database/SlimPajama-arrowed
nohup python -u slimpajama_packing.py\
    --dataset_size=5b\
    --print_interval=100 --num_process=200\
    --dataset_path=$PATH_TO_SLIMPAJAMA\
    --output_path=/data/home/scyb039/database/Splimpajama-DE/data/slimpajama/per_source_downsample --down_sample_ratio=0.1 --down_sample_mode=per_source\
    > /data/home/scyb039/database/Splimpajama-DE/logs/slimpajama_packing_dist_per_source_downsample_0.1.log 2>&1 &
tail -f /data/home/scyb039/database/Splimpajama-DE/logs/slimpajama_packing_dist_per_source_downsample_0.1.log