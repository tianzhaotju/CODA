## Preparing the Dataset
```shell
cd /root/Attack/CODA/AuthorshipAttribution/dataset/;

python get_reference.py \
    --all_data_file=../dataset/data_folder/processed_gcjpy/all.txt \
    --model_name=codebert;

python get_reference.py \
    --all_data_file=../dataset/data_folder/processed_gcjpy/all.txt \
    --model_name=graphcodebert;
```

## Adversarial Attack

```shell
cd /root/Attack/CODA/AuthorshipAttribution/code/;

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --model_name=codebert;
    
CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --model_name=graphcodebert;
```
