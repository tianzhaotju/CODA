## Preparing the Dataset
```shell
cd /root/Attack/CODA/FunctionalityClassification/dataset/;

python get_reference.py \
    --all_data_file=../dataset/all.txt \
    --model_name=codebert;

python get_reference.py \
    --all_data_file=../dataset/all.txt \
    --model_name=graphcodebert;
```

## Adversarial Attack
```shell
cd /root/Attack/CODA/FunctionalityClassification/code/;

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/test.txt \
    --model_name=codebert;
    
CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/test.txt \
    --model_name=graphcodebert;
```
