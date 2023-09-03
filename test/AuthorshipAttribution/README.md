## Preparing the Dataset
```shell
cd /root/CODA/test/AuthorshipAttribution/dataset/;

python get_reference.py \
    --all_data_file=../dataset/data_folder/processed_gcjpy/all.txt \
    --model_name=codebert;

python get_reference.py \
    --all_data_file=../dataset/data_folder/processed_gcjpy/all.txt \
    --model_name=graphcodebert;

```

## Testing

```shell
cd /root/CODA/test/AuthorshipAttribution/code/;

CUDA_VISIBLE_DEVICES=0 python test.py \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --model_name=codebert;
    
CUDA_VISIBLE_DEVICES=0 python test.py \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --model_name=graphcodebert;

CUDA_VISIBLE_DEVICES=0 python test.py \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --model_name=codet5;
```
