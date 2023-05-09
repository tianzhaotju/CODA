from __future__ import absolute_import, division, print_function
import os
import pickle
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm
import multiprocessing
from parser_folder import DFG_python, DFG_java, DFG_c
from run_parser import (remove_comments_and_docstrings, tree_to_token_index, index_to_code_token)
from tree_sitter import Language, Parser

cpu_cont = 16
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


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
    'c': DFG_c
}
#load parsers
parsers={}
for lang in dfg_function:
    LANGUAGE = Language('../../../python_parser/parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser,dfg_function[lang]]
    parsers[lang]= parser


def get_example(item):
    url1, url2, label, tokenizer, args, cache, url_to_code = item
    if url1 in cache:
        code1=cache[url1].copy()
    else:
        try:
            code=' '.join(url_to_code[url1].split())
        except:
            code=""
        code1=tokenizer.tokenize(code)
    if url2 in cache:
        code2=cache[url2].copy()
    else:
        try:
            code=' '.join(url_to_code[url2].split())
        except:
            code=""
        code2=tokenizer.tokenize(code)
    return codebert_convert_examples_to_features(code1,code2,label,url1,url2,tokenizer,args,cache)


def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code,'utf8'))
        root_node = tree.root_node
        tokens_index=tree_to_token_index(root_node)
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)
        try:
            DFG,_=parser[1](root_node,index_to_code,{})
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


class CodeBertInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, input_tokens, input_ids, label, url1, url2):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.url1=url1
        self.url2=url2


class GraphCodeBertInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, input_tokens_1, input_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1, input_tokens_2, input_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2, label, url1, url2):
        # The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1 = position_idx_1
        self.dfg_to_code_1 = dfg_to_code_1
        self.dfg_to_dfg_1 = dfg_to_dfg_1
        # The second code function
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2 = position_idx_2
        self.dfg_to_code_2 = dfg_to_code_2
        self.dfg_to_dfg_2 = dfg_to_dfg_2
        # label
        self.label = label
        self.url1 = url1
        self.url2 = url2


def codebert_convert_examples_to_features(code1_tokens,code2_tokens,label,url1,url2,tokenizer,args,cache):
    #source
    code1_tokens=code1_tokens[:args.block_size-2]
    code1_tokens =[tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens=code2_tokens[:args.block_size-2]
    code2_tokens =[tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]  
    
    code1_ids=tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids+=[tokenizer.pad_token_id]*padding_length
    
    code2_ids=tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids+=[tokenizer.pad_token_id]*padding_length
    
    source_tokens=code1_tokens+code2_tokens
    source_ids=code1_ids+code2_ids
    return CodeBertInputFeatures(source_tokens,source_ids,label,url1,url2)


def graphcodebert_convert_examples_to_features(item):
    # source
    url1, url2, label, tokenizer, args, cache, url_to_code = item
    parser = parsers['java']

    for url in [url1, url2]:
        if url not in cache:
            func = url_to_code[url]

            # extract data flow
            code_tokens, dfg = extract_dataflow(func, parser, 'java')
            code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                           enumerate(code_tokens)]
            ori2cur_pos = {}
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
            code_tokens = [y for x in code_tokens for y in x]

            # truncating
            code_tokens = code_tokens[
                          :args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][
                          :512 - 3]
            source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
            dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [tokenizer.unk_token_id for x in dfg]
            padding_length = args.code_length + args.data_flow_length - len(source_ids)
            position_idx += [tokenizer.pad_token_id] * padding_length
            source_ids += [tokenizer.pad_token_id] * padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([tokenizer.cls_token])
            dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
            cache[url] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

    source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1 = cache[url1]
    source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2 = cache[url2]
    return GraphCodeBertInputFeatures(source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1, source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2, label, url1, url2)


class CodeBertTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512,pool=None):
        postfix=file_path.split('/')[-1].split('.txt')[0]
        self.examples = []
        index_filename=file_path
        url_to_code={}
        folder = '/'.join(file_path.split('/')[:-1])

        cache_file_path = os.path.join(folder, '{}_cached_{}'.format(args.model_name, postfix))
        code_pairs_file_path = os.path.join(folder, '{}_cached_{}.pkl'.format(args.model_name, postfix))
        code_pairs = []
        try:
            self.examples = torch.load(cache_file_path)
            with open(code_pairs_file_path, 'rb') as f:
                code_pairs = pickle.load(f)
        except:
            if os.path.exists('/'.join(index_filename.split('/')[:-1])+'/adv_data.jsonl'):
                with open('/'.join(index_filename.split('/')[:-1])+'/adv_data.jsonl') as f:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        url_to_code[js['idx']] = js['func']
            else:
                with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        url_to_code[js['idx']] = js['func']
            data = []
            cache = {}
            with open(index_filename) as f:
                for line in f:
                    line = line.strip()
                    url1, url2, label = line.split('\t')
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    if label == '0':
                        label = 0
                    else:
                        label = 1
                    data.append((url1, url2, label, tokenizer, args, cache, url_to_code))
            for sing_example in data:
                code_pairs.append([sing_example[0], 
                                    sing_example[1], 
                                    url_to_code[sing_example[0]], 
                                    url_to_code[sing_example[1]]])
            with open(code_pairs_file_path, 'wb') as f:
                pickle.dump(code_pairs, f)
            pool = multiprocessing.Pool(7)
            self.examples=pool.map(get_example,tqdm(data,total=len(data)))
            torch.save(self.examples, cache_file_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)


class GraphCodeBertTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        postfix = file_path.split('/')[-1].split('.txt')[0]

        self.examples = []
        self.args = args
        index_filename = file_path

        # load index
        url_to_code = {}
        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, '{}_cached_{}'.format(args.model_name, postfix))
        code_pairs_file_path = os.path.join(folder, '{}_cached_{}.pkl'.format(args.model_name, postfix))
        code_pairs = []
        try:
            self.examples = torch.load(cache_file_path)
            with open(code_pairs_file_path, 'rb') as f:
                code_pairs = pickle.load(f)
        except:
            if os.path.exists('/'.join(index_filename.split('/')[:-1]) + '/adv_data.jsonl'):
                with open('/'.join(index_filename.split('/')[:-1]) + '/adv_data.jsonl') as f:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        url_to_code[js['idx']] = js['func']
            else:
                with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        url_to_code[js['idx']] = js['func']

            # load code function according to index
            data = []
            cache = {}
            with open(index_filename) as f:
                for line in f:
                    line = line.strip()
                    url1, url2, label = line.split('\t')
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    if label == '0':
                        label = 0
                    else:
                        label = 1
                    data.append((url1, url2, label, tokenizer, args, cache, url_to_code))

            # only use 10% valid data to keep best model
            for sing_example in data:
                code_pairs.append([sing_example[0],
                                   sing_example[1],
                                   url_to_code[sing_example[0]],
                                   url_to_code[sing_example[1]]])
            with open(code_pairs_file_path, 'wb') as f:
                pickle.dump(code_pairs, f)
            # convert example to input features
            self.examples = [graphcodebert_convert_examples_to_features(x) for x in tqdm(data, total=len(data))]
            torch.save(self.examples, cache_file_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask_1 = np.zeros((self.args.code_length + self.args.data_flow_length,
                                self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx_1])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_1])
        # sequence can attend to sequence
        attn_mask_1[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_1):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_1):
            if a < node_index and b < node_index:
                attn_mask_1[idx + node_index, a:b] = True
                attn_mask_1[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx_1):
                    attn_mask_1[idx + node_index, a + node_index] = True
                    # calculate graph-guided masked function
        attn_mask_2 = np.zeros((self.args.code_length + self.args.data_flow_length,
                                self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx_2])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_2])
        # sequence can attend to sequence
        attn_mask_2[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_2):
            if i in [0, 2]:
                attn_mask_2[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_2):
            if a < node_index and b < node_index:
                attn_mask_2[idx + node_index, a:b] = True
                attn_mask_2[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx_2):
                    attn_mask_2[idx + node_index, a + node_index] = True
        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1),
                torch.tensor(self.examples[item].input_ids_2),
                torch.tensor(self.examples[item].position_idx_2),
                torch.tensor(attn_mask_2),
                torch.tensor(self.examples[item].label))


def load_and_cache_examples(args, tokenizer, evaluate=False,test=False,pool=None):
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
