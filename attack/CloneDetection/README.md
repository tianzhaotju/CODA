## Attack
```shell
cd /root/Attack/CODA/CloneDetection/code/;

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/test_sampled_0_500.txt \
    --model_name=codebert;
    
CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/test_sampled_0_500.txt \
    --model_name=graphcodebert;
```
