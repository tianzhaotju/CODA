import os
import json
import sys
import copy
import torch
import argparse
from tqdm import tqdm
sys.path.append('../../../')
sys.path.append('../../../python_parser')
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings, get_example_batch
from utils import _tokenize
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
# sys.path.append('../')
# sys.path.append('../code/')
from model import CodeBERT, GraphCodeBERT
from run import CodeBertTextDataset, GraphCodeBertTextDataset
import numpy as np


MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


def get_embeddings(code, variables, tokenizer_mlm, codebert_mlm, args):
    new_code = copy.deepcopy(code)
    chromesome = {}
    for i in variables:
        chromesome[i] = '<unk>'
    new_code = get_example_batch(new_code, chromesome, "java")

    # for tgt_word in variables:
    #     new_code = get_example(new_code, tgt_word, '<unk>', "c")

    _, _, code_tokens = get_identifiers(remove_comments_and_docstrings(new_code, "java"), "java")
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
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")

    args = parser.parse_args()
    args.device = torch.device("cuda")
    # Set seed
    args.seed = 123456
    args.eval_batch_size = 32
    args.language_type = 'java'
    args.store_path = './%s_all_subs.json' % args.model_name
    args.n_gpu = 2
    args.block_size = 512

    if args.model_name == 'codebert':
        args.output_dir = '../code/saved_models'
        args.model_type = 'codebert_roberta'
        args.config_name = '/root/Attack/microsoft/codebert-base'
        args.model_name_or_path = '/root/Attack/microsoft/codebert-base'
        args.tokenizer_name = '/root/Attack/roberta-base'
        args.base_model = '/root/Attack/microsoft/codebert-base-mlm'
        args.number_labels = 2
    if args.model_name == 'graphcodebert':
        args.output_dir = '../code/saved_models'
        args.model_type = 'graphcodebert_roberta'
        args.config_name = '/root/Attack/microsoft/graphcodebert-base'
        args.tokenizer_name = '/root/Attack/microsoft/graphcodebert-base'
        args.model_name_or_path = '/root/Attack/microsoft/graphcodebert-base'
        args.base_model = '/root/Attack/microsoft/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64
        args.number_labels = 1

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
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

    checkpoint_prefix = 'checkpoint-best-f1/%s_model.bin' % (args.model_name)
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    # Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')

    url_to_code={}
    all_data = []
    with open('./data.jsonl') as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            url_to_code[js['idx']]=js['func']
    
    with open(args.all_data_file) as f:
        for i, line in enumerate(f):
            item = {}
            line=line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label=='0':
                label=0
                item["id1"] = url1
                item["id2"] = url2
                item["code1"] = url_to_code[url1]
                item["code2"] = url_to_code[url2]
                item["label"] = label
                all_data.append(item)
            else:
                label=1
                item["id1"] = url1
                item["id2"] = url2
                item["code1"] = url_to_code[url1]
                item["code2"] = url_to_code[url2]
                item["label"] = label
                all_data.append(item)
    # dict_keys(['id1', 'id2', 'code1', 'code2', 'label'])
    print(len(all_data))
    if args.model_name == 'codebert':
        all_examples = CodeBertTextDataset(tokenizer, args, args.all_data_file)
    elif args.model_name == 'graphcodebert':
        all_examples = GraphCodeBertTextDataset(tokenizer, args, args.all_data_file)

    assert len(all_examples) == len(all_data)
    all_labels = {}
    with open(args.store_path, "w") as wf:
        # for index in tqdm(range(len(all_data))):
        for index in tqdm(range(0, 15000)):
        # for index in tqdm(range(0, 150)):
            item = all_data[index]
            example = all_examples[index]

            logits, preds = model.get_results([example], args.eval_batch_size)

            if args.model_name == 'codebert':
                true_label = str(int(example[1].item()))
            elif args.model_name == 'graphcodebert':
                true_label = str(int(example[6].item()))
            orig_prob = np.max(logits[0])
            orig_label = str(int(preds[0]))

            if not true_label == orig_label:
                continue

            if true_label not in all_labels.keys():
                all_labels[true_label] = []

            code1 = item["code1"]
            code2 = item["code2"]

            variable_name1, function_name1, _ = get_identifiers(code1, "java")
            variable_name2, function_name2, _ = get_identifiers(code2, "java")

            variables1 = []
            variables1.extend(variable_name1)
            variables1.extend(function_name1)
            variables2 = []
            variables2.extend(variable_name2)
            variables2.extend(function_name2)

            embeddings1 = get_embeddings(code1, variables1, tokenizer_mlm, codebert_mlm, args)
            embeddings2 = get_embeddings(code2, variables2, tokenizer_mlm, codebert_mlm, args)

            if not os.path.exists('./%s_all_subs' % args.model_name):
                os.makedirs('./%s_all_subs' % args.model_name)
            np.save('./%s_all_subs/%s_%s_%s' % (args.model_name, str(orig_label), str(index), '1'), embeddings1.cpu().numpy())
            np.save('./%s_all_subs/%s_%s_%s' % (args.model_name, str(orig_label), str(index), '2'), embeddings2.cpu().numpy())
            all_labels[true_label].append({'code1': code1, 'code2': code2, 'embeddings_index': index,
                                           'variable_name1': variable_name1, 'variable_name2': variable_name2,
                                           'function_name1': function_name1, 'function_name2': function_name2})
        wf.write(json.dumps(all_labels) + '\n')


if __name__ == "__main__":
    main()
