## Attack

```shell
cd CODA/attack/AuthorshipAttribution/code/;

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --model_name=codebert;
    
CUDA_VISIBLE_DEVICES=0 python attack.py \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --model_name=graphcodeber;
```
