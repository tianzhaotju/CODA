from __future__ import absolute_import, division, print_function
import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
cpu_cont = 16
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from parser_folder import DFG_python, DFG_java, DFG_c
from run_parser import (remove_comments_and_docstrings, tree_to_token_index, index_to_code_token)
from tree_sitter import Language, Parser

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c,
}

#load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('../../../python_parser/parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    #remove comments
    code = code.replace("\\n", "\n")
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    #obtain dataflow
    if lang == "php":
        code = "<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x,code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx,code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class CodeBertInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


class GraphCodeBertInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, input_tokens, input_ids, position_idx, dfg_to_code, dfg_to_dfg, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.label = label


def codebert_convert_examples_to_features(code, label, tokenizer, args):
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return CodeBertInputFeatures(source_tokens, source_ids, 0, label)


def graphcodebert_convert_examples_to_features(code, label, tokenizer,args):
    parser = parsers["c"]
    code_tokens,dfg = extract_dataflow(code, parser, "c")

    code_tokens = [tokenizer.tokenize('@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1]+len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    code_tokens = code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg), args.data_flow_length)]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.code_length+args.data_flow_length-len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length+args.data_flow_length-len(source_ids)
    position_idx += [tokenizer.pad_token_id]*padding_length
    source_ids += [tokenizer.pad_token_id]*padding_length

    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]

    return GraphCodeBertInputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg,label)


class CodeBertTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        file_type = file_path.split('/')[-1].split('.')[0]
        folder = '/'.join(file_path.split('/')[:-1])

        cache_file_path = os.path.join(folder, '{}_cached_{}'.format(args.model_name, file_type))
        code_pairs_file_path = os.path.join(folder, '{}_cached_{}.pkl'.format(args.model_name, file_type))

        print('\n cached_features_file: ', cache_file_path)
        try:
            self.examples = torch.load(cache_file_path)
            with open(code_pairs_file_path, 'rb') as f:
                code_files = pickle.load(f)
        except:
            code_files = []
            with open(file_path) as f:
                for line in f:
                    code = line.split(" <CODESPLIT> ")[0]
                    code = code.replace("\\n", "\n").replace('\"', '"')
                    label = line.split(" <CODESPLIT> ")[1]
                    self.examples.append(codebert_convert_examples_to_features(code, int(label), tokenizer,args))
                    code_files.append(code)
            assert(len(self.examples) == len(code_files))
            with open(code_pairs_file_path, 'wb') as f:
                pickle.dump(code_files, f)
            torch.save(self.examples, cache_file_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)


class GraphCodeBertTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        self.args = args
        file_type = file_path.split('/')[-1].split('.')[0]
        folder = '/'.join(file_path.split('/')[:-1])

        cache_file_path = os.path.join(folder, '{}_cached_{}'.format(args.model_name, file_type))
        code_pairs_file_path = os.path.join(folder, '{}_cached_{}.pkl'.format(args.model_name, file_type))

        print('\n cached_features_file: ', cache_file_path)
        try:
            self.examples = torch.load(cache_file_path)
            with open(code_pairs_file_path, 'rb') as f:
                code_files = pickle.load(f)
        except:
            code_files = []
            with open(file_path) as f:
                for line in f:
                    code = line.split(" <CODESPLIT> ")[0]
                    code = code.replace("\\n", "\n").replace('\"', '"')
                    label = line.split(" <CODESPLIT> ")[1]
                    self.examples.append(graphcodebert_convert_examples_to_features(code, int(label), tokenizer, args))
                    code_files.append(code)
            assert (len(self.examples) == len(code_files))
            with open(code_pairs_file_path, 'wb') as f:
                pickle.dump(code_files, f)
            torch.save(self.examples, cache_file_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].label))


def codebert_load_and_cache_examples(args, tokenizer, evaluate=False, test=False, pool=None):
    dataset = CodeBertTextDataset(tokenizer, args, file_path=args.test_data_file if test else (args.eval_data_file if evaluate else args.train_data_file),block_size=args.block_size,pool=pool)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.model_name == 'codebert':
        os.environ['PYHTONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    elif args.model_name == 'graphcodebert':
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)