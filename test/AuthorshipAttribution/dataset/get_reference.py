import os
import json
import sys
import argparse
from tqdm import tqdm
sys.path.append('../../../')
sys.path.append('../../../python_parser')
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings, get_example_batch
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification)
from model import CodeBERT, GraphCodeBERT
from run import CodeBertTextDataset, GraphCodeBertTextDataset
import torch
import numpy as np
import copy
from utils import _tokenize


MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def get_embeddings(code, variables, tokenizer_mlm, codebert_mlm, args):
    new_code = copy.deepcopy(code)
    chromesome = {}
    for i in variables:
        chromesome[i] = '<unk>'
    new_code = get_example_batch(new_code, chromesome, "python")
    _, _, code_tokens = get_identifiers(remove_comments_and_docstrings(new_code, "python"), "python")
    processed_code = " ".join(code_tokens)
    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)
    sub_words = [tokenizer_mlm.cls_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.sep_token]
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    with torch.no_grad():
        embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")

    args = parser.parse_args()
    args.device = torch.device("cuda")
    args.seed = 123456
    args.block_size = 512
    args.eval_batch_size = 32
    args.number_labels = 66
    args.language_type = 'python'
    args.store_path = './data_folder/processed_gcjpy/%s_all_subs.json' % args.model_name

    if args.model_name == 'codebert':
        args.output_dir = '../code/saved_models'
        args.model_type = 'codebert_roberta'
        args.config_name = 'microsoft/codebert-base'
        args.model_name_or_path = 'microsoft/codebert-base'
        args.tokenizer_name = 'roberta-base'
        args.base_model = 'microsoft/codebert-base-mlm'
        args.block_size = 512
    if args.model_name == 'graphcodebert':
        args.output_dir = '../code/saved_models'
        args.model_type = 'graphcodebert_roberta'
        args.config_name = 'microsoft/graphcodebert-base'
        args.tokenizer_name = 'microsoft/graphcodebert-base'
        args.model_name_or_path = 'microsoft/graphcodebert-base'
        args.base_model = 'microsoft/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
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
    checkpoint_prefix = 'checkpoint-best-f1/%s_model.bin' % args.model_name
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    codebert_mlm.to(args.device)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)

    if args.model_name == 'codebert':
        all_dataset = CodeBertTextDataset(tokenizer, args, args.all_data_file)
    elif args.model_name == 'graphcodebert':
        all_dataset = GraphCodeBertTextDataset(tokenizer, args, args.all_data_file)
    source_codes = []
    with open(args.all_data_file) as rf:
        for line in rf:
            temp = line.replace("\\n", "\n").replace('\"', '"').split('<CODESPLIT>')[0]
            source_codes.append(temp)
    assert (len(source_codes) == len(all_dataset))
    print('length of all data', len(source_codes))

    all_labels = {}
    count = 0
    with open(args.store_path, "w") as wf:
        for index, example in tqdm(enumerate(all_dataset)):
            logits, preds = model.get_results([example], args.eval_batch_size)
            if args.model_name == 'codebert':
                true_label = str(int(example[1].item()))
            elif args.model_name == 'graphcodebert':
                true_label = str(int(example[3].item()))

            orig_prob = np.max(logits[0])
            orig_label = str(int(preds[0]))
            code = source_codes[index]

            if not true_label == orig_label:
                continue

            if true_label not in all_labels.keys():
                all_labels[true_label] = []

            try:
                variable_name, function_name, _ = get_identifiers(remove_comments_and_docstrings(code, "python"), "python")
            except:
                variable_name, function_name, _ = get_identifiers(code, "python")

            variables = []
            variables.extend(variable_name)
            variables.extend(function_name)

            embeddings = get_embeddings(code, variables, tokenizer_mlm, codebert_mlm, args)

            if not os.path.exists('./data_folder/processed_gcjpy/%s_all_subs' % args.model_name):
                os.makedirs('./data_folder/processed_gcjpy/%s_all_subs' % args.model_name)
            np.save('./data_folder/processed_gcjpy/%s_all_subs/%s_%s' % (args.model_name, str(orig_label), str(index)), embeddings.cpu().numpy())
            all_labels[true_label].append({'code': code, 'embeddings_index': index, 'variable_name': variable_name, 'function_name': function_name})
            count += 1
        print(count, len(all_dataset), count/len(all_dataset))
        wf.write(json.dumps(all_labels) + '\n')


if __name__ == "__main__":
    main()

