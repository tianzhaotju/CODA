import sys
sys.path.append('../../../')
sys.path.append('../../../python_parser')
import torch
import copy
from run import codebert_convert_examples_to_features, codet5_convert_examples_to_features, GraphCodeBertInputFeatures
import numpy as np
from utils import CodeDataset, CodePairDataset, CodeT5Dataset, _tokenize
from run_parser import get_identifiers, get_identifiers_ori, get_example, get_example_batch, get_code_style, \
    change_code_style, remove_comments_and_docstrings, extract_dataflow
from scipy.spatial.distance import cosine as cosine_distance


def graphcodebert_convert_code_to_features(code1, code2, tokenizer, label, args):
    feat = []
    for i, code in enumerate([code1, code2]):
        dfg, index_table, code_tokens = extract_dataflow(code, "java")

        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))
        code_tokens=[y for x in code_tokens for y in x]

        code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:args.code_length+args.data_flow_length-len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.code_length+args.data_flow_length-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length

        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
        feat.append((source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg))

    source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1=feat[0]
    source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2=feat[1]
    return GraphCodeBertInputFeatures(source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1,
                   source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2,
                     label, 0, 0)


def get_embeddings(code, variables, tokenizer_mlm, codebert_mlm):
    new_code = copy.deepcopy(code)
    chromesome = {}
    for i in variables:
        chromesome[i] = '<unk>'
    new_code = get_example_batch(new_code, chromesome, "java")
    _, _, code_tokens = get_identifiers(new_code, "java")
    processed_code = " ".join(code_tokens)
    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)
    sub_words = [tokenizer_mlm.cls_token] + sub_words[:512 - 2] + [tokenizer_mlm.sep_token]
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    with torch.no_grad():
        embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]

    return embeddings


class Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, codebert_mlm, fasttext_model, generated_substitutions) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.codebert_mlm = codebert_mlm
        self.fasttext_model = fasttext_model
        self.substitutions = generated_substitutions

    def attack(self, example, code_pair):
        NUMBER_1 = 256
        NUMBER_2 = 64
        code1 = code_pair[2]
        code2 = code_pair[3]
        invocation_number = 1
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)
        if self.args.model_name == 'codebert':
            true_label = example[1].item()
        elif self.args.model_name == 'graphcodebert':
            true_label = example[6].item()
        elif self.args.model_name == 'codet5':
            true_label = example[1].item()

        variable_names1, function_names1, _ = get_identifiers(code1, "java")
        variable_names2, function_names2, _ = get_identifiers(code2, "java")

        if (not orig_label == true_label) or len(variable_names1) == 0:
            return -2, None, None
        words2 = self.tokenizer_mlm.tokenize(" ".join(code2.split()))
        all_variable_name = []
        random_subs = []
        all_code = [code1] * NUMBER_2
        all_code_csc = [code1] * NUMBER_2
        orig_prob[np.argmax(orig_prob)] = -1
        topn_label = np.argmax(orig_prob)
        for i in np.random.choice(self.substitutions[str(topn_label)], size=len(self.substitutions[str(topn_label)]), replace=False):
            all_variable_name.extend(i['variable_name1'])
            all_variable_name.extend(i['function_name1'])
            if len(i['variable_name1']) < len(variable_names1):
                continue
            temp = copy.deepcopy(i)
            temp['label'] = str(topn_label)
            random_subs.append(temp)
            if len(random_subs) >= NUMBER_1:
                break
        substituions = []
        ori_embeddings1 = get_embeddings(code1, variable_names1+function_names1, self.tokenizer_mlm, self.codebert_mlm)
        ori_embeddings2 = get_embeddings(code2, [], self.tokenizer_mlm, self.codebert_mlm)
        ori_embeddings1 = torch.nn.functional.pad(ori_embeddings1, [0, 0, 0, 512 - np.shape(ori_embeddings1)[1]])
        ori_embeddings2 = torch.nn.functional.pad(ori_embeddings2, [0, 0, 0, 512 - np.shape(ori_embeddings2)[1]])

        embeddings_leng = np.shape(ori_embeddings1)[-1]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for sub in random_subs:
            embeddings_index = sub['embeddings_index']
            if self.args.model_name in ['codebert']:
                embeddings1 = np.load(
                    '../dataset/codebert_all_subs/%s_%s_%s.npy' % (sub['label'], embeddings_index, '1'))
            elif self.args.model_name in ['graphcodebert']:
                embeddings1 = np.load(
                    '../dataset/graphcodebert_all_subs/%s_%s_%s.npy' % (sub['label'], embeddings_index, '1'))
            elif self.args.model_name in ['codet5']:
                embeddings1 = np.load(
                    '../dataset/codebert_all_subs/%s_%s_%s.npy' % (sub['label'], embeddings_index, '1'))
            embeddings1 = torch.from_numpy(embeddings1).cuda()
            embeddings2 = get_embeddings(sub['code2'], [], self.tokenizer_mlm, self.codebert_mlm)
            embeddings1 = torch.nn.functional.pad(embeddings1, [0, 0, 0, 512 - np.shape(embeddings1)[1]])
            embeddings2 = torch.nn.functional.pad(embeddings2, [0, 0, 0, 512 - np.shape(embeddings2)[1]])
            cos_d = (np.sum(cos(ori_embeddings1, embeddings1).cpu().numpy()) / embeddings_leng) + np.sum(
                cos(ori_embeddings2, embeddings2).cpu().numpy()) / embeddings_leng
            substituions.append(([sub['variable_name1'], sub['function_name1'], sub['code1']], cos_d))

        substituions = sorted(substituions, key=lambda x: x[1], reverse=True)
        substituions = [x[0] for x in substituions[:NUMBER_2]]
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


        code_style = get_code_style(subs_code, 'java')
        replace_examples = []
        all_code_new = []
        for temp in all_code_csc:
            try:
                temp_code_ori = change_code_style(temp, "java", all_variable_name, code_style)[-1]
            except:
                temp_code_ori = temp
            all_code_new.append(temp_code_ori)
            temp_code = self.tokenizer_tgt.tokenize(" ".join(temp_code_ori.split()))
            if self.args.model_name == 'codebert':
                new_feature = codebert_convert_examples_to_features(temp_code,
                                                                    words2,
                                                                    example[1].item(),
                                                                    None, None,
                                                                    self.tokenizer_tgt,
                                                                    self.args, None)
            elif self.args.model_name == 'graphcodebert':
                new_feature = graphcodebert_convert_code_to_features(temp_code_ori,
                                                                     code2,
                                                                     self.tokenizer_tgt,
                                                                     example[6].item(),
                                                                     self.args)
            elif self.args.model_name == 'codet5':
                new_feature = codet5_convert_examples_to_features(temp_code,
                                                                  words2,
                                                                  example[1].item(),
                                                                  None, None,
                                                                  self.tokenizer_tgt,
                                                                  self.args, None)
            replace_examples.append(new_feature)

        if self.args.model_name == 'codebert':
            new_dataset = CodeDataset(replace_examples)
        elif self.args.model_name == 'graphcodebert':
            new_dataset = CodePairDataset(replace_examples, self.args)
        elif self.args.model_name == 'codet5':
            new_dataset = CodeT5Dataset(replace_examples)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

        for index, temp_prob in enumerate(logits):
            invocation_number += 1
            temp_label = preds[index]
            if temp_label != orig_label:
                print("%s SUCCESS! (%.5f => %.5f)" % ('>>', current_prob, temp_prob[orig_label]),
                      flush=True)
                return 2, all_code_new[index], current_prob - temp_prob[orig_label]
            else:
                if min_prob > temp_prob[orig_label]:
                    min_prob = temp_prob[orig_label]
                    code1 = all_code_new[index]
        print("%s FAIL! (%.5f => %.5f)" % ('>>', current_prob, min_prob), flush=True)

        subs_variable_name = []
        subs_function_name = []
        for i in temp_subs_variable_name:
            subs_variable_name.append([i, self.fasttext_model.get_word_vector(i)])
        for i in temp_subs_function_name:
            subs_function_name.append([i, self.fasttext_model.get_word_vector(i)])

        substituions = {}
        for i in variable_names1:
            temp = []
            i_vec = self.fasttext_model.get_word_vector(i)
            for j in subs_variable_name:
                if i == j[0]:
                    continue
                temp.append([j[0], 1 - cosine_distance(i_vec, j[1])])
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            substituions[i] = [x[0] for x in temp]
        for i in function_names1:
            temp = []
            i_vec = self.fasttext_model.get_word_vector(i)
            for j in subs_function_name:
                if i == j[0]:
                    continue
                temp.append([j[0], 1 - cosine_distance(i_vec, j[1])])
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            substituions[i] = [x[0] for x in temp]
        replace_examples = []
        all_code = []
        all_code_csc = []
        current_subs = ['' for i in range(len(variable_names1) + len(function_names1))]

        for i in range(NUMBER_2):
            temp_code = copy.deepcopy(code1)
            for j, tgt_word in enumerate(variable_names1):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j] = substituions[tgt_word][i]
                temp_code = get_example(temp_code, tgt_word, substituions[tgt_word][i], "java")
                temp_code_ori = copy.deepcopy(temp_code)
                all_code.append(temp_code)
                temp_code = self.tokenizer_tgt.tokenize(" ".join(temp_code.split()))
                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_examples_to_features(temp_code,
                                                                        words2,
                                                                        example[1].item(),
                                                                        None, None,
                                                                        self.tokenizer_tgt,
                                                                        self.args, None)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(temp_code_ori,
                                                                         code2,
                                                                         self.tokenizer_tgt,
                                                                         example[6].item(),
                                                                         self.args)
                elif self.args.model_name == 'codet5':
                    new_feature = codet5_convert_examples_to_features(temp_code,
                                                                      words2,
                                                                      example[1].item(),
                                                                      None, None,
                                                                      self.tokenizer_tgt,
                                                                      self.args, None)
                replace_examples.append(new_feature)
                temp_code = temp_code_ori

            for j, tgt_word in enumerate(function_names1):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j + len(variable_names1)] = substituions[tgt_word][i]
                temp_code = get_example(temp_code, tgt_word, substituions[tgt_word][i], "java")
                temp_code_ori = copy.deepcopy(temp_code)
                all_code.append(temp_code)
                temp_code = self.tokenizer_tgt.tokenize(" ".join(temp_code.split()))

                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_examples_to_features(temp_code,
                                                                        words2,
                                                                        example[1].item(),
                                                                        None, None,
                                                                        self.tokenizer_tgt,
                                                                        self.args, None)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(temp_code_ori,
                                                                         code2,
                                                                         self.tokenizer_tgt,
                                                                         example[6].item(),
                                                                         self.args)
                elif self.args.model_name == 'codet5':
                    new_feature = codet5_convert_examples_to_features(temp_code,
                                                                      words2,
                                                                      example[1].item(),
                                                                      None, None,
                                                                      self.tokenizer_tgt,
                                                                      self.args, None)
                replace_examples.append(new_feature)
                temp_code = temp_code_ori
            all_code_csc.append(all_code[-1])

        if len(replace_examples) == 0:
            return -3, None, None

        if self.args.model_name == 'codebert':
            new_dataset = CodeDataset(replace_examples)
        elif self.args.model_name == 'graphcodebert':
            new_dataset = CodePairDataset(replace_examples, self.args)
        elif self.args.model_name == 'codet5':
            new_dataset = CodeT5Dataset(replace_examples)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

        final_code = None
        for index, temp_prob in enumerate(logits):
            invocation_number += 1
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