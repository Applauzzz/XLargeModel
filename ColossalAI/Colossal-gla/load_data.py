from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import glob
import os
import tempfile
# os.environ['HF_DATASETS_CACHE'] = '/mnt/nvme_storage/tmpfile/datasets'
# os.environ['HF_HOME'] = '/mnt/nvme_storage/tmpfile'
# os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme_storage/tmpfile'

# datapath = "/gpfsdata/yuhong/database/Infinity-Instruct/3M"
datapath = "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow"
save_dir = "/mnt/nvme_storage/Zehao/ColossalAI/zehao_model/database/inf-ins-3M-jsonl"

# tempfile.tempdir = '/mnt/nvme_storage/tmpfile'
# # 创建保存目录
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# from datasets import Dataset

# # 读取指定路径下的 .arrow 文件
# dirs = glob.glob(os.path.join(datapath, "*"))
# dirs.sort()
# dirs = dirs[:1]
# print(dirs)
# for dir in dirs:
#     arrow_files = glob.glob(os.path.join(dir, "*.arrow"))
#     arrow_files.sort()
#     print(arrow_files)
#     for arrow_file in arrow_files:
#         dataset = Dataset.from_file(arrow_file)
#         mmax = 0
#         for input in dataset:
#             mmax = max(mmax, input['seq_length']) 
#         print(mmax)
#         print(dataset[0])
#         print("raw::")

 ############load stage1/2 data################

# train_file_path = "/mnt/nvme_storage/database/Infinity-instruct-tokenized4gla/arrow"
train_file_path = "/data/home/scyb039/database/SlimPj_mistral_4_CoAI_const_packing_all/arrow/part-00000"
arrow_files = []
# tokenized_data_path = os.path.join(train_file_path, "stage1_data_new2")
print(f"tokenized_data_path: {train_file_path}")
# dirs = os.listdir(train_file_path)


arrow_files += glob.glob(os.path.join(train_file_path, "*.arrow"))
print(len(arrow_files))
arrow_files = arrow_files
train_dataset_stage1 = load_dataset('arrow', data_files=arrow_files, split='train')
# train_dataset_stage1 = train_dataset_stage1.filter(lambda x: len(x["input_ids"]) != 8192)
# train_dataset_stage1 = train_dataset_stage1.shuffle(seed=42)
# print("len:",len(train_dataset_stage1))
# # train_dataset_stage1.sort("conversation_id")
# print(train_dataset_stage1[0].keys())
# print(len(train_dataset_stage1[0]))
# print(len(train_dataset_stage1[0]["input_ids"]))
# # print(train_dataset_stage1[0]["input_ids"])
# print(len(train_dataset_stage1)*2048*10/10000000000)
print(len(train_dataset_stage1.filter(lambda x: len(x["input_ids"]) != 8192, num_proc=24)))
tokenizer = AutoTokenizer.from_pretrained("/data/home/scyb039/modelbase/Mistral-7B-v0.1", use_fast=False)
# print(tokenizer.decode(train_dataset_stage1[0]["input_ids"]))
# # logger.info(f"stage1 data size: {len(train_dataset_stage1)} records, batch size = 1024, steps = {len(train_dataset_stage1) / 1024}")
# arrow_files = []
# tokenized_data_path = os.path.join(args.train_file, "stage2_data_new2")
# logger.info(f"tokenized_data_path: {tokenized_data_path}")
# for dir in dirs:
#     # print(os.path.join(tokenized_data_path, dir, "*.arrow"))
#     arrow_files += glob.glob(os.path.join(tokenized_data_path, dir, "*.arrow"))
# train_dataset_stage2 = load_dataset('arrow', data_files=arrow_files, split='train')
# train_dataset_stage2 = train_dataset_stage2.shuffle(seed=42)
# # logger.info(f"stage2 data size: {len(train_dataset_stage2)}")
# train_dataset = concatenate_datasets([train_dataset_stage1, train_dataset_stage2])
# # train_dataset = train_dataset_stage2
# # logger.info(f"total data size: {len(train_dataset)}")
# logger.info(f"stage1 data size: {len(train_dataset_stage1)} records, batch size = 1024, steps = {len(train_dataset_stage1) / 1024}\n stage2 data size: {len(train_dataset_stage2)} records, batch size = 1024, steps = {len(train_dataset_stage2) / 1024}\n total data size: {len(train_dataset)}, batch size = 1024, steps = {len(train_dataset) / 1024}\n")
        ############################################

# 获取所有 Parquet 文件
# parquet_files = glob.glob(os.path.join(datapath, "*.parquet"))
# parquet_files.sort()
# # 遍历每个 Parquet 文件并分别转换为 JSONL
# for parquet_file in parquet_files:
#     # 获取文件名
#     filename = os.path.basename(parquet_file)
    
#     # 读取每个 Parquet 文件
#     df = pd.read_parquet(parquet_file)
    
#     # 确保按原始顺序处理（若有指定排序的字段）
#     # df = df.sort_values(by=['column_name'])  # 如果有排序字段的话
#     for i in range(3):
#         print(parquet_file)
#         print(df['id'][i])
#     # for i in range(3):
#     #     print(parquet_file)
#     #     print(df['id'][-3+i])
#     # 生成对应的 JSONL 文件名
#     jsonl_file = filename.replace('.parquet', '.jsonl')
    
#     # 保存为 JSONL 文件（确保按原始顺序写入）
#     df.to_json(os.path.join(save_dir, jsonl_file), orient="records", lines=True)
    
#     # 验证记录数是否一致
#     ct1 = len(df)
#     with open(os.path.join(save_dir, jsonl_file), 'r') as f:
#         ct2 = len(f.readlines())
#         print(f"验证记录数: {ct1} == {ct2}: {ct1 == ct2}")
    
#     print(f"{parquet_file} 已转换为 {jsonl_file}")


# # 读取 JSONL 文件
# jsonl_files = glob.glob(os.path.join(save_dir, "*.jsonl"))
# jsonl_files.sort()
# #打印一条记录
# for ind in range(1):
#     print(jsonl_files[ind])
#     # 不要重排打乱
#     ds = load_dataset("json", data_files = {'train':jsonl_files[ind]})
#     # ds = sorted(ds['train'], key=lambda x: x['id'])
#     # ds.filter(lambda data: data["labels"] is not None)
#     for i in range(1):
#         print(ds['train'][14350])
#         # print(ds[i]['id'])
#     # for i in range(3):
#     #     print(ds['train'][-3+i]['id'])
#     #     print(ds[i]['conversations'])

# ds = load_dataset(
# "parquet",
# data_files = {'train':f'{datapath}/*.parquet'
# }
# )

# tokenizer = AutoTokenizer.from_pretrained("/gpfsdata/yuhong/ColossalAI/zehao_model/mistral")
# model_max_length = 1000000000000000019884624838656
# #print top 10 examples
# # for i in range(10):
# #     print(ds["train"][i])
# def tokenize_and_filter(e):
#     try:
#         tokens = tokenizer(
#             " ".join([conv['value'] for conv in e['conversations'] if isinstance(conv, dict) and 'value' in conv and conv['value']]),
#             truncation=True,
#             padding=model_max_length,
#             max_length=model_max_length
#         )
#         if len(tokens['input_ids']) == model_max_length:
#             return tokens
#         else:
#             print(f"Skipping due to wrong length: {len(tokens['input_ids'])}")
#             return None
#     except Exception as ex:
#         # print(f"Error processing: {e}, Error: {ex}")
#         return None

# tokenized_ds = ds.map(
#     tokenize_and_filter,
#     batched=True,
#     remove_columns=ds['train'].column_names
# )


# tokenized_ds.save_to_disk("/gpfsdata/yuhong/ColossalAI/zehao_model/database/infinity-instruct-3M_tokenized")

# 300000
# 399999
# 399998
# 399997
# 399996
# 399995
# 399994
# 399993
# 399992
# 399991
# Generating train split: 100000 examples [00:00, 211019.72 examples/s]
# 500000
# 599999
# 599998
# 599997
# 599996
# 599995
# 599994
# 599993
# 599992
# 599991
# 2100000
# 2199999
# 2199998
# 2199997
# 2199996
# 2199995
# 2199994
# 2199993
# 2199992
# 2199991
# Generating train split: 100000 examples [00:00, 199367.34 examples/s]
# 2300000
# 2399999
# 2399998
# 2399997
# 2399996
# 2399995
# 2399994
# 2399993
# 2399992
# 2399991
# Generating train split: 100000 examples [00:00, 230505.66 examples/s]
# 600000
# 699999
# 699998
# 699997
# 699996
# 699995
# 699994
# 699993
# 699992
# 699991
# Generating train split: 100000 examples [00:00, 210057.49 examples/s]
# 3000000
# 3099999
# 3099998
# 3099997
# 3099996
# 3099995
# 3099994
# 3099993
# 3099992
# 3099991
# Generating train split: 63473 examples [00:00, 216661.03 examples/s]
# 3400000
# 3463472
# 3463471
# 3463470
# 3463469
# 3463468
# 3463467
# 3463466
# 3463465
# 3463464
# Generating train split: 100000 examples [00:00, 240350.24 examples/s]
# 1500000
# 1599999
# 1599998
# 1599997
# 1599996
# 1599995
# 1599994
# 1599993
# 1599992
# 1599991
# Generating train split: 100000 examples [00:00, 229433.77 examples/s]
# 1100000
# 1199999
# 1199998
# 1199997
# 1199996
# 1199995
# 1199994
# 1199993
# 1199992
# 1199991
# 2400000
# 2499999
# 2499998
# 2499997
# 2499996
# 2499995
# 2499994
# 2499993
# 2499992
# 2499991