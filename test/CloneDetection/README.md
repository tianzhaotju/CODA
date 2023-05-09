## Preparing the Dataset
```shell
cd /root/Attack/CODA/CloneDetection/dataset/;

python get_reference.py \
    --all_data_file=../dataset/all.txt \
    --model_name=codebert;

python get_reference.py \
    --all_data_file=../dataset/all.txt \
    --model_name=graphcodebert;
```

## Test
```shell
cd /root/Attack/CODA/CloneDetection/code/;

CUDA_VISIBLE_DEVICES=0 python test.py \
    --eval_data_file=../dataset/test_sampled_0_500.txt \
    --model_name=codebert;
    
CUDA_VISIBLE_DEVICES=0 python test.py \
    --eval_data_file=../dataset/test_sampled_0_500.txt \
    --model_name=graphcodebert;

CUDA_VISIBLE_DEVICES=0 python test.py \
    --eval_data_file=../dataset/test_sampled_0_500.txt \
    --model_name=codet5;
```
