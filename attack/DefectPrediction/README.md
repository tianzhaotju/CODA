## Attack
```shell
cd /root/Attack/CODA/DefectPrediction/code/;

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/test.txt \
    --model_name=codebert;
    
CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/test.txt \
    --model_name=graphcodebert;
```
