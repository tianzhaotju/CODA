import sys
sys.path.append('../../../')
sys.path.append('../../../python_parser')
import torch
import copy
from run import CodeT5InputFeatures, CodeBertInputFeatures, GraphCodeBertInputFeatures, extract_dataflow
import numpy as np
from utils import CodeDataset, GraphCodeDataset, CodeT5Dataset, _tokenize
from run_parser import get_identifiers, get_example, get_example_batch, get_code_style, change_code_style, remove_comments_and_docstrings
from scipy.spatial.distance import cosine as cosine_distance
from parser_folder import DFG_python, DFG_java, DFG_c
from tree_sitter import Language, Parser
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c
}

parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('../../../python_parser/parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def codebert_convert_code_to_features(code, tokenizer, label, args):
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return CodeBertInputFeatures(source_tokens, source_ids, 0, label)


def graphcodebert_convert_code_to_features(code, tokenizer, label, args):
    parser = parsers["python"]
    code_tokens, dfg = extract_dataflow(code, parser, "python")
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
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
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    return GraphCodeBertInputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label)


def codet5_convert_code_to_features(code, tokenizer, label, args):
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return CodeT5InputFeatures(source_tokens, source_ids, 0, label)


def get_embeddings(code, variables, tokenizer_mlm, codebert_mlm):
    new_code = copy.deepcopy(code)
    chromesome = {}
    for i in variables:
        chromesome[i] = '<unk>'
    new_code = get_example_batch(new_code, chromesome, "python")
    _, _, code_tokens = get_identifiers(new_code, "python")
    processed_code = " ".join(code_tokens)
    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)
    sub_words = [tokenizer_mlm.cls_token] + sub_words[:512 - 2] + [tokenizer_mlm.sep_token]
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    with torch.no_grad():
        embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]
    return embeddings


class Attacker:
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, codebert_mlm, fasttext_model, generated_substitutions) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.codebert_mlm = codebert_mlm
        self.fasttext_model = fasttext_model
        self.substitutions = generated_substitutions

    def attack(self, example, code):
        NUMBER_1 = 256
        NUMBER_2 = 64
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)
        if self.args.model_name == 'codebert':
            true_label = example[1].item()
        elif self.args.model_name == 'graphcodebert':
            true_label = example[3].item()
        elif self.args.model_name == 'codet5':
            true_label = example[1].item()
        variable_names, function_names, code_tokens = get_identifiers(code, "python")
        if (not orig_label == true_label) or len(variable_names)+len(function_names) == 0:
            return -2, None, None
        all_variable_name = []
        random_subs = []
        all_code = [code] * NUMBER_2
        all_code_csc = [code] * NUMBER_2

        while len(random_subs) < NUMBER_1 and np.max(orig_prob) >= 0:
            orig_prob[np.argmax(orig_prob)] = -1
            topn_label = np.argmax(orig_prob)
            for i in np.random.choice(self.substitutions[str(topn_label)], size=len(self.substitutions[str(topn_label)]), replace=False):
                if len(i['variable_name']) < len(variable_names) or len(i['function_name']) < len(function_names):
                    continue
                all_variable_name.extend(i['variable_name'])
                temp = copy.deepcopy(i)
                temp['label'] = str(topn_label)
                random_subs.append(temp)
                if len(random_subs) >= NUMBER_1:
                    break

        substituions = []
        ori_embeddings = get_embeddings(code, variable_names+function_names, self.tokenizer_mlm, self.codebert_mlm)
        ori_embeddings = torch.nn.functional.pad(ori_embeddings, [0, 0, 0, 512 - np.shape(ori_embeddings)[1]])

        embeddings_leng = np.shape(ori_embeddings)[-1]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for sub in random_subs:
            embeddings_index = sub['embeddings_index']
            if self.args.model_name in ['codebert']:
                embeddings = np.load('../dataset/data_folder/processed_gcjpy/codebert_all_subs/%s_%s.npy' % (
                sub['label'], embeddings_index))
            elif self.args.model_name in ['graphcodebert']:
                embeddings = np.load('../dataset/data_folder/processed_gcjpy/graphcodebert_all_subs/%s_%s.npy' % (
                sub['label'], embeddings_index))
            elif self.args.model_name in ['codet5']:
                embeddings = np.load('../dataset/data_folder/processed_gcjpy/codebert_all_subs/%s_%s.npy' % (
                sub['label'], embeddings_index))
            embeddings = torch.from_numpy(embeddings).cuda()
            embeddings = torch.nn.functional.pad(embeddings, [0, 0, 0, 512 - np.shape(embeddings)[1]])
            substituions.append(([sub['variable_name'], sub['function_name'], sub['code']],
                                 np.sum(cos(ori_embeddings, embeddings).cpu().numpy()) / embeddings_leng))

        substituions = sorted(substituions, key=lambda x: x[1], reverse=True)
        substituions = [x[0] for x in substituions[:NUMBER_2]]
        max_number = len(substituions)
        temp_subs_variable_name = set()
        temp_subs_function_name = set()
        subs_code = []
        for subs in substituions:
            for i in subs[0]:
                temp_subs_variable_name.add(i)
            for i in subs[1]:
                temp_subs_function_name.add(i)
            subs_code.append(subs[2])
        min_prob = current_prob

        all_code_new = []
        code_style = get_code_style(subs_code, 'python')
        replace_examples = []
        for temp in all_code_csc:
            try:
                temp_code = change_code_style(temp, "python", all_variable_name, code_style)[-1]
            except:
                temp_code = temp
            all_code_new.append(temp_code)
            if self.args.model_name == 'codebert':
                new_feature = codebert_convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
            elif self.args.model_name == 'graphcodebert':
                new_feature = graphcodebert_convert_code_to_features(temp_code, self.tokenizer_tgt, example[3].item(), self.args)
            elif self.args.model_name == 'codet5':
                new_feature = codet5_convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
            replace_examples.append(new_feature)

        if self.args.model_name == 'codebert':
            new_dataset = CodeDataset(replace_examples)
        elif self.args.model_name == 'graphcodebert':
            new_dataset = GraphCodeDataset(replace_examples, self.args)
        elif self.args.model_name == 'codet5':
            new_dataset = CodeT5Dataset(replace_examples)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

        for index, temp_prob in enumerate(logits):
            temp_label = preds[index]
            if temp_label != orig_label:
                print("%s SUCCESS! (%.5f => %.5f)" % ('>>', current_prob, temp_prob[orig_label]),
                      flush=True)
                return 2, all_code_new[index], current_prob - min(min_prob, temp_prob[orig_label])
            else:
                if min_prob > temp_prob[orig_label]:
                    min_prob = temp_prob[orig_label]
                    code = all_code_new[index]
        print("%s FAIL! (%.5f => %.5f)" % ('>>', current_prob, min_prob), flush=True)

        subs_variable_name = []
        subs_function_name = []
        for i in temp_subs_variable_name:
            subs_variable_name.append([i, self.fasttext_model.get_word_vector(i)])
        for i in temp_subs_function_name:
            subs_function_name.append([i, self.fasttext_model.get_word_vector(i)])
        substituions = {}
        for i in variable_names:
            temp = []
            i_vec = self.fasttext_model.get_word_vector(i)
            for j in subs_variable_name:
                if i == j[0]:
                    continue
                temp.append([j[0], 1 - cosine_distance(i_vec, j[1])])
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            substituions[i] = [x[0] for x in temp]
        for i in function_names:
            temp = []
            i_vec = self.fasttext_model.get_word_vector(i)
            for j in subs_function_name:
                if i == j[0]:
                    continue
                temp.append([j[0], 1 - cosine_distance(i_vec, j[1])])
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            substituions[i] = [x[0] for x in temp]
        all_code = []
        all_code_csc = []
        replace_examples = []
        current_subs = ['' for i in range(len(variable_names) + len(function_names))]
        for i in range(max_number):
            temp_code = copy.deepcopy(code)
            for j, tgt_word in enumerate(variable_names):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j] = substituions[tgt_word][i]
                temp_code = get_example(temp_code, tgt_word, substituions[tgt_word][i], "python")
                all_code.append(temp_code)
                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                    example[1].item(), self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                         example[3].item(), self.args)
                elif self.args.model_name == 'codet5':
                    new_feature = codet5_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                    example[1].item(), self.args)
                replace_examples.append(new_feature)
            for j, tgt_word in enumerate(function_names):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j + len(variable_names)] = substituions[tgt_word][i]
                temp_code = get_example(temp_code, tgt_word, substituions[tgt_word][i], "python")

                all_code.append(temp_code)
                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                    example[1].item(), self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                         example[3].item(), self.args)
                elif self.args.model_name == 'codet5':
                    new_feature = codet5_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                  example[1].item(), self.args)
                replace_examples.append(new_feature)
            all_code_csc.append(all_code[-1])

        if len(replace_examples) == 0:
            return -3, None, None
        if self.args.model_name == 'codebert':
            new_dataset = CodeDataset(replace_examples)
        elif self.args.model_name == 'graphcodebert':
            new_dataset = GraphCodeDataset(replace_examples, self.args)
        elif self.args.model_name == 'codet5':
            new_dataset = CodeT5Dataset(replace_examples)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
        final_code = None
        for index, temp_prob in enumerate(logits):
            temp_label = preds[index]
            if temp_label != orig_label:
                print("%s SUCCESS! (%.5f => %.5f)" % ('>>', current_prob, temp_prob[orig_label]),
                      flush=True)
                return 1, all_code[index], current_prob - temp_prob[orig_label]
            else:
                if min_prob >= temp_prob[orig_label]:
                    min_prob = temp_prob[orig_label]
                    final_code = all_code[index]
        print("%s FAIL! (%.5f => %.5f)" % ('>>', current_prob, min_prob), flush=True)

        return -1, final_code, current_prob - min_prob