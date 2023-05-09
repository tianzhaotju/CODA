import sys
import os
sys.path.append('../../../')
sys.path.append('../../../python_parser')
retval = os.getcwd()
import json
import argparse
import warnings
import torch
from run import set_seed
from run import CodeBertTextDataset, GraphCodeBertTextDataset, CodeT5TextDataset
from model import CodeBERT, GraphCodeBERT, CodeT5
from attacker import Attacker
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaModel, RobertaTokenizer,
                          RobertaForSequenceClassification, T5Config, T5ForConditionalGeneration)
import fasttext
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning\
MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")

    args = parser.parse_args()
    args.device = torch.device("cuda")
    # Set seed
    args.seed = 123456
    args.number_labels = 66
    args.eval_batch_size = 32
    args.language_type = 'python'
    args.n_gpu = 2
    args.block_size = 512
    args.use_ga = True

    if args.model_name == 'codebert':
        args.output_dir = './saved_models'
        args.model_type = 'codebert_roberta'
        args.config_name = 'microsoft/codebert-base'
        args.model_name_or_path = 'microsoft/codebert-base'
        args.tokenizer_name = 'roberta-base'
        args.base_model = 'microsoft/codebert-base-mlm'
    if args.model_name == 'graphcodebert':
        args.output_dir = './saved_models'
        args.model_type = 'graphcodebert_roberta'
        args.config_name = 'microsoft/graphcodebert-base'
        args.tokenizer_name = 'microsoft/graphcodebert-base'
        args.model_name_or_path = 'microsoft/graphcodebert-base'
        args.base_model = 'microsoft/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64
    if args.model_name == 'codet5':
        args.output_dir = './saved_models'
        args.model_type = 'codet5'
        args.config_name = 'Salesforce/codet5-base-multi-sum'
        args.tokenizer_name = 'Salesforce/codet5-base-multi-sum'
        args.model_name_or_path = 'Salesforce/codet5-base-multi-sum'
        args.base_model = 'microsoft/codebert-base-mlm'
    set_seed(args)
    ## Load Target Model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_name_or_path == 'codebert':
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, 510)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    if args.model_name == 'codebert':
        model = CodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'graphcodebert':
        model = GraphCodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'codet5':
        model = CodeT5(model, config, tokenizer, args)
    checkpoint_prefix = 'checkpoint-best-f1/%s_model.bin' % args.model_name
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))      
    model.to(args.device)
    # load fasttext
    fasttext_model = fasttext.load_model("../../../fasttext_model.bin")
    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')
    generated_substitutions = json.load(open('../dataset/data_folder/processed_gcjpy/%s_all_subs.json' % (args.model_name), 'r'))
    attacker = Attacker(args, model, tokenizer, tokenizer_mlm, codebert_mlm, fasttext_model, generated_substitutions)
    source_codes = []
    with open(args.eval_data_file) as rf:
        for line in rf:
            source_codes.append(line.split(' <CODESPLIT> ')[0].strip().replace("\\n", "\n").replace('\"', '"'))

    ## Load Dataset
    if args.model_name == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'codet5':
        eval_dataset = CodeT5TextDataset(tokenizer, args, args.eval_data_file)
    print(len(eval_dataset), len(source_codes))
    success_attack = 0
    total_cnt = 0
    for index, example in enumerate(eval_dataset):
        # if index in [6, 36, 37]:
        #     continue
        code = source_codes[index]
        is_success, final_code, min_gap_prob = attacker.attack(
            example,
            code,
            identifier=True,
            structure=True
        )
        if is_success >= -1:
            total_cnt += 1
            if is_success >= 1:
                success_attack += 1
            if total_cnt == 0:
                continue
            print("Success rate: %.2f%%" % ((1.0 * success_attack / total_cnt) * 100))
            print("Successful items count: ", success_attack)
            print("Total count: ", total_cnt)
            print("Index: ", index)
            print()


if __name__ == '__main__':
    main()

