
CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt --model_name=codebert
CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt --model_name=graphcodebert

CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/test_sampled_0_500.txt --model_name=codebert
CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/test_sampled_0_500.txt --model_name=graphcodebert

CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/dataset/test_subs_0_400.jsonl --model_name=codebert
CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/dataset/test_subs_0_400.jsonl --model_name=graphcodebert

CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/test.txt --model_name=codebert
CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/test.txt --model_name=graphcodebert

CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/test.txt --model_name=codebert
CUDA_VISIBLE_DEVICES=0 python attack.py --eval_data_file=../dataset/test.txt --model_name=graphcodebert