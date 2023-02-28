import os.path
import random
import click
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
import math
from nltk import ngrams
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, \
    BertModel
import torch.nn as nn
import csv
import numpy as np
import re
from scipy.stats import poisson
import time
from transformers import MBartForConditionalGeneration, MBartTokenizer
import glob
from custom_operator_utils import select_random_synonym, lemmatize
from nltk import word_tokenize
import re
import string
from mqm_preprocess import preprocess_wmt_data
import argparse


def adjust_punctuation(sen_list):
    new_list = []
    for sen in sen_list:
        new_sen = []
        for i, char in enumerate(sen):
            if char.isalnum() or (char == "'" and sen[i - 1].isalnum() and sen[i + 1].isalnum()):
                new_sen.append(char)
            elif char in [',', '.']:
                while i > 0 and new_sen[-1] == ' ':
                    new_sen.pop(-1)
                new_sen.append(char)
            else:
                new_sen.append(' ')
        new_sen = ''.join(new_sen)
        new_list.append(" ".join(new_sen.split()))
    return new_list


"""Yield batch sized list of sentences."""


def parse_pmi(owt=False):
    filename = f"PMI/pmi-{'owt-' if owt else ''}wiki-bc_clean.txt"
    with open(filename) as f:
        data = f.read().split("\n")
        result = sorted([(len(row), row) for row in data], key=lambda x: x[0], reverse=True)
        result = set([unit for _, unit in result])
    return result


def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i + batch_size, len(lst))]


def noise_sanity_check(cand_arr, num_noises, del_noise_lam=None, mask_noise_lam=None, word_tokens=True,
                       use_new_operators=False):
    # decide noise type upon function called, only sentences have one noise and step 1 can have MBart noises
    if num_noises == 1:
        # TODO: change the if pmi: statements to include yoni's operators, currently if PMI is enabled then yoni'ws operators receive no weight
        if word_tokens and use_new_operators:
            noise_type = \
                random.choices([1, 2, 3, 4, 5, 6, 7, 8],
                               weights=(1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8), k=1)[0]
        else:
            noise_type = \
                random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=(1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 0), k=1)[
                    0]
    else:
        if word_tokens and use_new_operators:
            noise_type = random.choices([3, 4, 5, 6, 7, 8], weights=(1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6), k=1)[0]
        else:
            noise_type = random.choices([3, 4, 5, 6, 7, 8], weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4, 0, 0), k=1)[0]

    if noise_type == 1 or noise_type == 2:
        start_index = random.choices(range(cand_arr['mbart'].shape[0]), k=1)[0]
    else:
        start_index = random.choices(range(cand_arr['xlm'].shape[0]), k=1)[0]
    # this is the MBart addition noise
    if noise_type == 1:
        # check if noise position and span length fits current noise context
        if cand_arr['mbart'][start_index] > 0:
            return noise_type, start_index, 0
    # this is the MBart replace noise which can replace the span of noises
    elif noise_type == 2:
        if mask_noise_lam:
            num_replace = \
                random.choices([1, 2, 3, 4, 5, 6], weights=poisson.pmf(np.arange(1, 7, 1), mu=mask_noise_lam, loc=1),
                               k=1)[0]
        else:
            num_replace = random.choices([1, 2, 3, 4, 5, 6], k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr['mbart'][start_index] >= num_replace and cand_arr['mbart'][start_index] != 0:
            return noise_type, start_index, num_replace
    # this is the XLM addition noise
    elif noise_type == 3:
        if cand_arr['xlm'][start_index] > 0 and cand_arr['xlm'][start_index] != 0:
            return noise_type, start_index, 0
    # this is the XLM replace noise
    elif noise_type == 4:
        if cand_arr['xlm'][start_index] > 0 and cand_arr['xlm'][start_index] != 0:
            return noise_type, start_index, 1
    # this is the swap noise
    elif noise_type == 5:
        if cand_arr['xlm'].shape[0] > 6:
            # within range 4 choose the second index, -5 because we are switching start_index+1 and end_index+1
            start_index = random.choices(range(cand_arr['xlm'].shape[0] - 6), k=1)[
                0]  # indices = sorted(random.sample(range(cand_arr['xlm'].shape[0]), 2))
            end_index = start_index + random.choices([1, 2, 3, 4], k=1)[
                0]  # start_index, end_index = indices[0], indices[1]
            if cand_arr['xlm'][start_index] > 0 and cand_arr['xlm'][end_index] > 0:
                return noise_type, start_index, end_index
    # this is the delete noise
    elif noise_type == 6:
        if del_noise_lam:
            num_deletes = \
                random.choices([1, 2, 3, 4], weights=poisson.pmf(np.arange(1, 5, 1), mu=del_noise_lam, loc=1), k=1)[0]
        else:
            num_deletes = random.choices([1, 2, 3, 4], k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr['xlm'][start_index] >= num_deletes and cand_arr['xlm'][start_index] != 0:
            return noise_type, start_index, num_deletes

    elif noise_type == 7:
        if cand_arr['xlm'][start_index] > 0:
            return noise_type, start_index, 1

    else:
        if cand_arr['xlm'][start_index] > 0:
            return noise_type, start_index, 1
    return -1, -1, -1


"""return planned noise combinations for each sentence with num_var variances"""


def noise_planner(num_var, num_texts, lam):
    sen_noise_dict = {}
    max_step = 0
    for sen_index in range(num_texts):
        for noise_index in range(num_var):
            # Random selection of number of noises
            num_noises = random.choices([1, 2, 3, 4, 5], weights=poisson.pmf(np.arange(1, 6, 1), mu=lam, loc=1), k=1)[0]
            sen_noise_dict[str(sen_index) + '_' + str(noise_index)] = num_noises
            if num_noises > max_step:
                max_step = num_noises
    return sen_noise_dict, max_step  # return in dict: key->segID_noiseID, value->num of noises (A list of noise types)


"""seq list dict: key is step index, value is a dict of sentences: key is the segID_noiseID. value is the modifed sentence
    dict: key->segID_noiseID, value->[original sentence, noise1, noise2, ... noisek]"""


def noise_schedule(id_sen_dict, step, sen_noise_dict, cand_dict_arr, del_noise_lam, mask_noise_lam, word_tokens=True,
                   use_new_operators=False):
    # all the necessary information to construct all 6 noise types
    mbart_add_seg_id_ls, mbart_add_start_ls = [], []
    mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls = [], [], []
    xlm_add_seg_id_ls, xlm_add_start_ls = [], []
    xlm_replace_seg_id_ls, xlm_replace_start_ls = [], []
    swap_seg_id_ls, swap_start_ls, swap_end_ls = [], [], []
    del_seg_id_ls, del_start_ls, del_len_ls = [], [], []
    xlm_synonym_seg_id_ls, xlm_synonym_start_ls = [], []
    xlm_lemmatization_seg_id_ls, xlm_lemmatization_start_ls = [], []
    step_noise_dict = {}
    # check if the given noise type and condition is valid and return the valid noise and condition
    for id, num_noises in sen_noise_dict.items():
        # check if the segment has the valid number of noise for current step
        if step <= num_noises:
            noise_type, start_index, num_ops = noise_sanity_check(cand_dict_arr[id], num_noises, del_noise_lam,
                                                                  mask_noise_lam, word_tokens, use_new_operators)
            # only if random selected error type and error number is valid
            if noise_type != -1:
                # type1: MBart Addition noise
                if noise_type == 1:  # store mbart add start index and corresponding seg id
                    mbart_add_seg_id_ls.append(id)
                    mbart_add_start_ls.append(start_index)
                    step_noise_dict[id] = ['MBart Addition', start_index]
                # type2: MBart replace noise
                elif noise_type == 2:
                    mbart_replace_seg_id_ls.append(id)
                    mbart_replace_start_ls.append(start_index)
                    mbart_replace_len_ls.append(num_ops)
                    step_noise_dict[id] = ['MBart Replace', start_index, num_ops]
                # type3: XLM-Roberta Addition
                elif noise_type == 3:
                    xlm_add_seg_id_ls.append(id)
                    xlm_add_start_ls.append(start_index)
                    step_noise_dict[id] = ['XLM Addition', start_index]
                # type4: XLM-Roberta Replace
                elif noise_type == 4:
                    xlm_replace_seg_id_ls.append(id)
                    xlm_replace_start_ls.append(start_index)
                    step_noise_dict[id] = ['XLM Replace', start_index, 1]
                # type6: Swap noise
                elif noise_type == 5:
                    swap_seg_id_ls.append(id)
                    swap_start_ls.append(start_index)
                    swap_end_ls.append(num_ops)
                    step_noise_dict[id] = ['Switch', start_index, num_ops]
                elif noise_type == 6:  # Accuracy/Omission, Fluency/Grammar
                    del_seg_id_ls.append(id)
                    del_start_ls.append(start_index)
                    del_len_ls.append(num_ops)
                    step_noise_dict[id] = ['Delete', start_index, num_ops]
                elif noise_type == 7:
                    xlm_synonym_seg_id_ls.append(id)
                    xlm_synonym_start_ls.append(start_index)
                    step_noise_dict[id] = ['Synonyms_replace', start_index, 1]
                else:
                    xlm_lemmatization_seg_id_ls.append(id)
                    xlm_lemmatization_start_ls.append(start_index)
                    step_noise_dict[id] = ['lemmatization_replace', start_index, 1]

    # seg_id_ls: a list contains all the seg_noise ids
    return mbart_add_seg_id_ls, mbart_add_start_ls, mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls, xlm_add_seg_id_ls, xlm_add_start_ls, \
           xlm_replace_seg_id_ls, xlm_replace_start_ls, swap_seg_id_ls, swap_start_ls, swap_end_ls, del_seg_id_ls, del_start_ls, del_len_ls, xlm_synonym_seg_id_ls, \
           xlm_synonym_start_ls, xlm_lemmatization_seg_id_ls, xlm_lemmatization_start_ls, step_noise_dict


"""add operation to update the candidate dict"""


def add_update_cand_dict(cand_dict_arr, add_seg_id_ls, add_start_ls):
    for add_seg_id, add_start in zip(add_seg_id_ls, add_start_ls):
        new_cand_ls = []
        # update all the values on the left
        for index, _ in enumerate(cand_dict_arr[add_seg_id]['xlm'][:add_start + 1]):
            new_cand_ls.append(min(add_start - index, cand_dict_arr[add_seg_id]['xlm'][index]))
        new_cand_ls.extend([0])
        if cand_dict_arr[add_seg_id]['xlm'].shape[
            0] == 126:  # xlm at most have sequence length of 126 to incorporate <s> </s> tokens
            new_cand_ls.extend(list(cand_dict_arr[add_seg_id]['xlm'][add_start + 1:-1]))
        cand_dict_arr[add_seg_id]['xlm'] = np.array(new_cand_ls)
    return cand_dict_arr


"""replace operation to update the candidate dict"""


def replace_update_cand_dict(cand_dict_arr, replace_seg_id_ls, replace_start_ls):
    for replace_seg_id, replace_start in zip(replace_seg_id_ls, replace_start_ls):
        new_cand_ls = []
        # update all the values on the left
        for index, _ in enumerate(cand_dict_arr[replace_seg_id]['xlm'][:replace_start + 1]):
            new_cand_ls.append(min(replace_start - index, cand_dict_arr[replace_seg_id]['xlm'][index]))
        new_cand_ls.extend([0])
        new_cand_ls.extend(list(cand_dict_arr[replace_seg_id]['xlm'][replace_start + 2:]))
        cand_dict_arr[replace_seg_id]['xlm'] = np.array(new_cand_ls)
    return cand_dict_arr


"""delete operation to update the candidate dict"""


def delete_update_cand_dict(cand_dict_arr, del_seg_id_ls, del_start_ls, del_len_ls):
    for del_seg_id, del_start, del_len in zip(del_seg_id_ls, del_start_ls, del_len_ls):
        new_cand_ls = []
        # update all the values on the left
        for index in range(len(cand_dict_arr[del_seg_id]['xlm'][:del_start + 1])):
            new_cand_ls.append(min(del_start - index, cand_dict_arr[del_seg_id]['xlm'][index]))
        new_cand_ls.extend(list(cand_dict_arr[del_seg_id]['xlm'][del_start + 1 + del_len:]))
        cand_dict_arr[del_seg_id]['xlm'] = np.array(new_cand_ls)
    return cand_dict_arr


def prev_ids_sens_extract(id_sen_dict, new_seg_ids):
    prev_sen_ls = []
    for id in new_seg_ids:
        prev_sen_ls.append(id_sen_dict[id]['text'][-1])
    return prev_sen_ls


def swap_update_cand_dict(cand_dict_arr, swap_seg_id_ls, swap_start_ls, swap_end_ls):
    for swap_seg_id, swap_start, swap_end in zip(swap_seg_id_ls, swap_start_ls, swap_end_ls):
        new_cand_ls = []
        for index in range(len(cand_dict_arr[swap_seg_id]['xlm'][:swap_start + 1])):
            new_cand_ls.append(min(swap_start - index, cand_dict_arr[swap_seg_id]['xlm'][index]))
        new_cand_ls += [0]
        for index in range(len(cand_dict_arr[swap_seg_id]['xlm'][swap_start + 2:swap_end + 1])):
            new_cand_ls.append(
                min(swap_end - swap_start - index, cand_dict_arr[swap_seg_id]['xlm'][swap_start + 2 + index]))
        new_cand_ls += [0] + list(cand_dict_arr[swap_seg_id]['xlm'][swap_end + 2:])
        cand_dict_arr[swap_seg_id]['xlm'] = np.array(new_cand_ls)
    return cand_dict_arr


def severity_measure(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device):
    with torch.no_grad():
        inputs_1 = mnli_tokenizer(prev_batch, cur_batch, return_tensors="pt", max_length=256, truncation=True,
                                  padding=True).to(device)
        output_1 = mnli_model(**inputs_1).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_1 = m(output_1)[:, -1]

        inputs_2 = mnli_tokenizer(cur_batch, prev_batch, return_tensors="pt", max_length=256, truncation=True,
                                  padding=True).to(device)
        output_2 = mnli_model(**inputs_2).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_2 = m(output_2)[:, -1]

    # Use the harmonic mean for threshold, threshold = 0.9
    # p_not_severe = softmax_result_1/(1-softmax_result_1) * softmax_result_2/(1-softmax_result_2)
    scores = []
    for prob_1, prob_2 in zip(softmax_result_1, softmax_result_2):
        if prob_1 > 0.9 and prob_2 > 0.9:
            scores.append(-1)
        else:
            scores.append(-5)
    return scores


def severity_measure_with_positive_correction(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device,
                                              original_sen, segs, lam=1,
                                              delta=1, analysis_logger=None):
    with torch.no_grad():
        inputs_1 = mnli_tokenizer(prev_batch, cur_batch, return_tensors="pt", max_length=256, truncation=True,
                                  padding=True).to(device)
        output_1 = mnli_model(**inputs_1).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_1 = m(output_1)[:, -1]

        inputs_2 = mnli_tokenizer(cur_batch, prev_batch, return_tensors="pt", max_length=256, truncation=True,
                                  padding=True).to(device)
        output_2 = mnli_model(**inputs_2).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_2 = m(output_2)[:, -1]

        original_sen = [original_sen[int(idx.split("_")[0])] for idx in segs]
        inputs_org_prev_1 = mnli_tokenizer(original_sen, prev_batch, return_tensors="pt", max_length=256,
                                           truncation=True,
                                           padding=True).to(device)
        output_org_prev_1 = mnli_model(**inputs_org_prev_1).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_org_prev_1 = m(output_org_prev_1)[:, -1]

        inputs_org_prev_2 = mnli_tokenizer(prev_batch, original_sen, return_tensors="pt", max_length=256,
                                           truncation=True,
                                           padding=True).to(device)
        output_org_prev_2 = mnli_model(**inputs_org_prev_2).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_org_prev_2 = m(output_org_prev_2)[:, -1]

        inputs_org_curr_1 = mnli_tokenizer(original_sen, cur_batch, return_tensors="pt", max_length=256,
                                           truncation=True,
                                           padding=True).to(device)
        output_org_curr_1 = mnli_model(**inputs_org_curr_1).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_org_curr_1 = m(output_org_curr_1)[:, -1]

        inputs_org_curr_2 = mnli_tokenizer(cur_batch, original_sen, return_tensors="pt", max_length=256,
                                           truncation=True,
                                           padding=True).to(device)
        output_org_curr_2 = mnli_model(**inputs_org_curr_2).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_org_curr_2 = m(output_org_curr_2)[:, -1]

    # Use the harmonic mean for threshold, threshold = 0.9
    # p_not_severe = softmax_result_1/(1-softmax_result_1) * softmax_result_2/(1-softmax_result_2)
    scores = []

    for i, (prob_1, prob_2, pi_10, p0i_1, pi0, p0i) in enumerate(
            zip(softmax_result_1, softmax_result_2, softmax_result_org_prev_1,
                softmax_result_org_prev_2, softmax_result_org_curr_1,
                softmax_result_org_curr_2)):
        ratio1 = pi0 / pi_10
        ratio2 = p0i / p0i_1
        cond1 = ratio1 > delta and ratio2 > delta
        if cond1 and analysis_logger is not None:
            analysis_logger.write(original_sen[i] + '\n')
            analysis_logger.write(prev_batch[i] + '\n')
            analysis_logger.write(cur_batch[i] + '\n')
            analysis_logger.flush()
        cond2 = prob_1 > 0.9 and prob_2 > 0.9
        scores.append((-1 if cond2 else -5) + lam * (1 if cond1 else 0))
    return scores


def severity_measure_with_continuous_scoring_geometric_mean(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device, n,
                                             analysis_logger):
    with torch.no_grad():
        inputs_1 = mnli_tokenizer(prev_batch, cur_batch, return_tensors="pt", max_length=256, truncation=True,
                                  padding=True).to(device)
        output_1 = mnli_model(**inputs_1).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_1 = m(output_1)[:, -1]

        inputs_2 = mnli_tokenizer(cur_batch, prev_batch, return_tensors="pt", max_length=256, truncation=True,
                                  padding=True).to(device)
        output_2 = mnli_model(**inputs_2).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_2 = m(output_2)[:, -1]

    # Use the harmonic mean for threshold, threshold = 0.9
    # p_not_severe = softmax_result_1/(1-softmax_result_1) * softmax_result_2/(1-softmax_result_2)
    scores = []

    for i, (prob_1, prob_2) in enumerate(zip(softmax_result_1, softmax_result_2)):
        bde = (prob_1.item()*prob_2.item())**0.5
        score = bde**n - 1
        if analysis_logger is not None:
            analysis_logger.write(f"The continuous score was {score} \n")
            analysis_logger.write(prev_batch[i] + '\n')
            analysis_logger.write(cur_batch[i] + '\n')
            analysis_logger.flush()
        scores.append(score)
    return scores

def severity_measure_with_continuous_scoring_arithmetic_mean(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device, n,
                                             analysis_logger):
    with torch.no_grad():
        inputs_1 = mnli_tokenizer(prev_batch, cur_batch, return_tensors="pt", max_length=256, truncation=True,
                                  padding=True).to(device)
        output_1 = mnli_model(**inputs_1).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_1 = m(output_1)[:, -1]

        inputs_2 = mnli_tokenizer(cur_batch, prev_batch, return_tensors="pt", max_length=256, truncation=True,
                                  padding=True).to(device)
        output_2 = mnli_model(**inputs_2).logits  # 0: contradiction, 1: neutral, 2: entailment
        softmax_result_2 = m(output_2)[:, -1]

    # Use the harmonic mean for threshold, threshold = 0.9
    # p_not_severe = softmax_result_1/(1-softmax_result_1) * softmax_result_2/(1-softmax_result_2)
    scores = []

    for i, (prob_1, prob_2) in enumerate(zip(softmax_result_1, softmax_result_2)):
        bde = (prob_1.item() + prob_2.item())/2
        score = bde**n - 1
        if analysis_logger is not None:
            analysis_logger.write(f"The continuous score was {score} \n")
            analysis_logger.write(prev_batch[i] + '\n')
            analysis_logger.write(cur_batch[i] + '\n')
            analysis_logger.flush()
        scores.append(score)
    return scores

# The reason we work this way (include the word in this calculation) is because we want all the word
# tokens to be included.Then we remove 1 in order to work with the code.
# If we worked with the word before, we don't know how many tokens the next word is.
def infer_token_location_index_mbart(text, tokenizer, index, pmi):
    words = text.split()
    sub_sen = '</s> ' + " ".join(words[:index + 1])
    if pmi:
        sub_sen = sub_sen.replace("♣", " ")
    # tok_text = \
    #     tokenizer(sub_sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)[
    #         'input_ids'][0]
    tok_text = \
        tokenizer(sub_sen, add_special_tokens=True, return_tensors="pt", truncation=True, padding=True)[
            'input_ids'][0]
    # End tokens of </s> and en_XX
    new_start_index = len(tok_text) - 1 - 2
    return new_start_index


def infer_ops_num_and_start_index_mbart(text, tokenizer, start_index, num_ops, pmi):
    token_start = infer_token_location_index_mbart(text, tokenizer, start_index, pmi)
    words = text.split()
    sub_sen = '</s> ' + " ".join(words[:start_index + num_ops + 1])
    if pmi:
        sub_sen = sub_sen.replace("♣", " ")
    # tok_text = \
    #     tokenizer(sub_sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)[
    #         'input_ids'][0]
    tok_text = \
        tokenizer(sub_sen, add_special_tokens=True, return_tensors="pt", truncation=True, padding=True)[
            'input_ids'][0]
    new_ops = len(tok_text) - token_start - 1 - 2
    return token_start, new_ops


def infer_token_location_index_xlm(text, tokenizer, index, pmi):
    words = text.split()
    sub_sen = " ".join(words[:index + 1])
    if pmi:
        sub_sen = sub_sen.replace("♣", " ")
    # tok_text = \
    #     tokenizer(sub_sen, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
    #               padding=True)['input_ids'][0]
    tok_text = \
        tokenizer(sub_sen, add_special_tokens=False, return_tensors="pt", truncation=True,
                  padding=True)['input_ids'][0]
    new_start_index = len(tok_text)
    return new_start_index


def infer_ops_num_and_start_index_xlm(text, tokenizer, start_index, num_ops, pmi):
    token_start = infer_token_location_index_xlm(text, tokenizer, start_index, pmi)
    words = text.split()
    sub_sen = " ".join(words[:start_index + num_ops + 1])
    if pmi:
        sub_sen = sub_sen.replace("♣", " ")
    # tok_text = \
    #     tokenizer(sub_sen, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
    #               padding=True)['input_ids'][0]
    tok_text = \
        tokenizer(sub_sen, add_special_tokens=False, return_tensors="pt", truncation=True,
                  padding=True)['input_ids'][0]
    new_ops = len(tok_text) - token_start
    return token_start, new_ops


def infer_swap_size(text, tokenizer, start_index, num_ops, pmi):
    words = text.split()
    first_word = words[start_index + 1]
    second_word = words[num_ops + 1]
    if pmi:
        first_word = first_word.replace("♣", " ")
        second_word = second_word.replace("♣", " ")
    # tok_text_first = \
    #     tokenizer(first_word, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
    #               padding=True)['input_ids'][0]
    tok_text_first = \
        tokenizer(first_word, add_special_tokens=False, return_tensors="pt", truncation=True,
                  padding=True)['input_ids'][0]
    # tok_text_second = \
    #     tokenizer(second_word, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
    #               padding=True)['input_ids'][0]
    tok_text_second = \
        tokenizer(second_word, add_special_tokens=False, return_tensors="pt", truncation=True,
                  padding=True)['input_ids'][0]
    return len(tok_text_first), len(tok_text_second)


def data_construct(text, noise_type, tokenizer, start_index, num_ops, word_tokens=True, pmi=True, analysis_logger=None):
    if not word_tokens:
        start_index += 1  # to incorporate beginning token
    if noise_type == 1:
        if word_tokens:
            start_index = infer_token_location_index_mbart(text, tokenizer, start_index, pmi)
        sen = '</s> ' + text
        if pmi:
            sen = sen.replace("♣", " ")
        # tok_text = \
        #     tokenizer(sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True,
        #               padding=True)[
        #         'input_ids']
        tok_text = \
            tokenizer(sen, add_special_tokens=True, return_tensors="pt", truncation=True,
                      padding=True)['input_ids']
        input_ids = torch.cat(
            (tok_text[0][:start_index + 2], torch.LongTensor([tokenizer.mask_token_id]),
             tok_text[0][start_index + 2:]),
            dim=0)  # index shifts by 1 bc of </s>
    elif noise_type == 2:
        if word_tokens:
            start_index, num_ops = infer_ops_num_and_start_index_mbart(text, tokenizer, start_index, num_ops, pmi)
        sen = '</s> ' + text
        if pmi:
            sen = sen.replace("♣", " ")
        # tok_text = \
        #     tokenizer(sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)[
        #         'input_ids']
        tok_text = \
            tokenizer(sen, add_special_tokens=True, return_tensors="pt", truncation=True, padding=True)[
                'input_ids']
        input_ids = torch.cat((tok_text[0][:start_index + 2], torch.LongTensor([tokenizer.mask_token_id]),
                               tok_text[0][start_index + 2 + num_ops:]), dim=0)  # index shifts by 1 bc of </s>
    elif noise_type == 3:
        if word_tokens:
            start_index = infer_token_location_index_xlm(text, tokenizer, start_index, pmi)
        if pmi:
            text = text.replace("♣", " ")
        # tok_text = \
        #     tokenizer(text, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
        #               padding=True)['input_ids']
        tok_text = \
            tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=True,
                      padding=True)['input_ids']
        input_ids = torch.cat(
            (tok_text[0][:start_index + 1], torch.LongTensor([tokenizer.mask_token_id]), tok_text[0][start_index + 1:]),
            dim=0)
    elif noise_type == 4:
        if word_tokens:
            start_index, num_ops = infer_ops_num_and_start_index_xlm(text, tokenizer, start_index, num_ops, pmi)
        if pmi:
            text = text.replace("♣", " ")
        # tok_text = \
        #     tokenizer(text, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
        #               padding=True)['input_ids']
        tok_text = \
            tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=True,
                      padding=True)['input_ids']
        input_ids = torch.cat((tok_text[0][:start_index + 1], torch.LongTensor([tokenizer.mask_token_id]),
                               tok_text[0][start_index + 1 + num_ops:]), dim=0)
    elif noise_type == 5:
        if word_tokens:
            first_word_length, second_word_length = infer_swap_size(text, tokenizer, start_index, num_ops, pmi)
            start_index = infer_token_location_index_xlm(text, tokenizer, start_index, pmi)
            num_ops = infer_token_location_index_xlm(text, tokenizer, num_ops, pmi)
        end_index = num_ops
        if pmi:
            text = text.replace("♣", " ")
        # tok_text = \
        #     tokenizer(text, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
        #               padding=True)['input_ids']
        tok_text = \
            tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=True,
                      padding=True)['input_ids']
        if word_tokens:
            input_ids = torch.cat((tok_text[0][:start_index],
                                   tok_text[0][end_index: end_index + second_word_length],
                                   tok_text[0][start_index + first_word_length:end_index],
                                   tok_text[0][start_index:start_index + first_word_length],
                                   tok_text[0][end_index + second_word_length:]), dim=0)

        else:
            input_ids = torch.cat((tok_text[0][:start_index + 1],
                                   torch.unsqueeze(tok_text[0][end_index + 1], 0),
                                   tok_text[0][start_index + 2:end_index + 1],
                                   torch.unsqueeze(tok_text[0][start_index + 1], 0),
                                   tok_text[0][end_index + 2:]), dim=0)
        if word_tokens:
            new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            return adjust_punctuation([new_text])[0]
        else:
            return tokenizer.decode(input_ids, skip_special_tokens=True)

    elif noise_type == 6:
        if word_tokens:
            start_index, num_ops = infer_ops_num_and_start_index_xlm(text, tokenizer, start_index, num_ops, pmi)
        if pmi:
            text = text.replace("♣", " ")
        # tok_text = \
        #     tokenizer(text, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
        #               padding=True)[
        #         'input_ids']
        tok_text = \
            tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=True,
                      padding=True)[
                'input_ids']
        input_ids = torch.cat((tok_text[0][:start_index + 1], tok_text[0][start_index + 1 + num_ops:]), dim=0)
        if word_tokens:
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            return adjust_punctuation([text])[0]
    elif noise_type == 7:
        tokens = text.split()
        if analysis_logger is not None:
            temp = select_random_synonym(tokens[start_index])
            if temp == tokens[start_index]:
                analysis_logger.write('Synonym_ failed \n')
            else:
                analysis_logger.write('Synonym_ succeeded \n')
            tokens[start_index] = temp
        else:
            tokens[start_index] = select_random_synonym(tokens[start_index])
        text = " ".join(tokens)
        text = adjust_punctuation([text])[0]
        return text
    else:
        words = text.split()
        if analysis_logger is not None:
            temp = lemmatize(words[start_index])
            if temp == words[start_index]:
                analysis_logger.write('Lemmatization_ failed \n')
            else:
                analysis_logger.write('Lemmatization_ succeeded \n')
            words[start_index] = temp
        else:
            words[start_index] = lemmatize(words[start_index])
        text = ' '.join(words)
        text = adjust_punctuation([text])[0]
        return text

    return tokenizer.decode(input_ids, skip_special_tokens=False)


def mbart_generation(batch_text, model, lang_code, tokenizer, device, word_tokens):
    with torch.no_grad():
        batch = tokenizer(batch_text, return_tensors="pt", max_length=128, truncation=True, padding=True)[
            'input_ids'].to(device)

        translated_tokens = model.generate(batch, decoder_start_token_id=tokenizer.lang_code_to_id[lang_code])
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        if word_tokens:
            return adjust_punctuation(translation)
        return translation
        # input_ids = \
        #     tokenizer(batch_text, add_special_tokens=True, return_tensors='pt', max_length=128, truncation=True,
        #                   padding=True)['input_ids'].to(device)
        # input_ids = \
        #     tokenizer(batch_text, add_special_tokens=True, return_tensors='pt', truncation=True,
        #               padding=True)['input_ids'].to(device)
        # logits = model(input_ids).logits
        #
        # for i, ele_input_ids in enumerate(input_ids):
        #     try:
        #         masked_index = (ele_input_ids == tokenizer.mask_token_id).nonzero().item()
        #     except ValueError:
        #         print(batch_text[i])
        #         print(ele_input_ids)
        #         print(tokenizer.mask_token_id)
        #     probs = logits[i, masked_index].softmax(dim=0)
        #     values, predictions = probs.topk(4)
        #     pred = random.choices(predictions, k=1)[0]
        #     input_ids[i][masked_index] = pred
        # return tokenizer.batch_decode(input_ids, skip_special_tokens=True)


def xlm_roberta_generate(batch_text, model, xlm_tokenizer, device, word_tokens):
    with torch.no_grad():  # need to add special tokens bc previous steps didn't
        # input_ids = \
        #     xlm_tokenizer(batch_text, add_special_tokens=True, return_tensors='pt', max_length=128, truncation=True,
        #                   padding=True)['input_ids'].to(device)
        input_ids = \
            xlm_tokenizer(batch_text, add_special_tokens=True, return_tensors='pt', truncation=True,
                          padding=True)['input_ids'].to(device)
        logits = model(input_ids).logits

        for i, ele_input_ids in enumerate(input_ids):
            try:
                masked_index = (ele_input_ids == xlm_tokenizer.mask_token_id).nonzero().item()
            except ValueError:
                print(batch_text[i])
                print(ele_input_ids)
                print(xlm_tokenizer.mask_token_id)
                print(i)
            probs = logits[i, masked_index].softmax(dim=0)
            values, predictions = probs.topk(4)
            pred = random.choices(predictions, k=1)[0]
            input_ids[i][masked_index] = pred
        text = xlm_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        if word_tokens:
            return adjust_punctuation(text)
        return text
        # return xlm_tokenizer.batch_decode(input_ids, skip_special_tokens=True)


# def text_score_generate(num_var, lang, ref_lines, noise_planner_num, del_noise_lam, mask_noise_lam, device,
#                         severity='original', word_tokens=False, pmi=False,use_new_operators = False,mbart_batch_size=8,xlm_batch_size = 128
#                         ,mnli_batch_size=128 ):
def text_score_generate(num_var, lang, ref_lines, noise_planner_num, del_noise_lam, mask_noise_lam, device, args,
                        log_file, dir_path):
    # load in XLM-Roberta model
    severity = args.severity
    word_tokens = args.word_tokens
    pmi = args.pmi
    mnli_batch_size = args.mnli_batch_size
    mbart_batch_size = args.mbart_batch_size
    xlm_batch_size = args.xlm_batch_size
    use_new_operators = args.use_new_operators
    analysis_logger = None
    if args.analysis_mode:
        analysis_save_dir_run = args.analysis_save_dir +'/'+ args.save_name_start +'_operators'
        if not os.path.exists(analysis_save_dir_run):
            os.mkdir(analysis_save_dir_run)
        else:
            raise Exception("Error in analysis save path, it already exists")
        analysis_logger = open(analysis_save_dir_run+'/analysis.txt', 'w')
    xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    xlm_model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-large')
    xlm_model.eval()
    # load in MBart model and its tokenzier
    mbart_model = MBartForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path="facebook/mbart-large-cc25")
    mbart_tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang=lang)
    mbart_model.eval()
    # initialize cand_dict_arr, sen_noise_dict, id_sen_dict: key->seg_noise id, value->sentence list
    cand_dict_arr = {}
    id_sen_score_dict = {}
    id_sen_dict = {}  # id_sen_dict is a dict containing "score" and "text" fields, "text" field is a list which contains a history of all generated sentences
    pmi_order = parse_pmi(False)
    for line_index, ref_line in tqdm(enumerate(ref_lines)):
        for i in range(num_var):
            id = str(line_index) + '_' + str(i)
            if word_tokens:
                words = ref_line.split()
                if pmi:
                    used_indecies = [0 for _ in range(len(words))]
                    for n in range(4, 1, -1):
                        ngrams_lst = ngrams(ref_line.split(), n)
                        for k, gram in enumerate(ngrams_lst):
                            gram_txt = " ".join(gram)
                            flag = not any([used_indecies[j] for j in range(k, k + n)])
                            if gram_txt in pmi_order and flag:
                                for j in range(k, k + n):
                                    used_indecies[j] = 1
                                gram_txt_replace = gram_txt.replace(" ", "♣")
                                ref_line = ref_line.replace(gram_txt, gram_txt_replace)
                words = ref_line.split()
                tok_xlm_ls = words
                tok_mbart_ls = words
            else:
                tok_xlm_ls = xlm_tokenizer.tokenize(ref_line)
                tok_mbart_ls = mbart_tokenizer.tokenize(ref_line)
            # initialize pretraining scheduling scheme using tokenized word lists
            cand_dict_arr[id] = {}
            cand_dict_arr[id]['xlm'] = min(len(tok_xlm_ls), 126) - 1 - np.array(range(min(len(tok_xlm_ls), 126)))
            cand_dict_arr[id]['mbart'] = min(len(tok_mbart_ls), 125) - 1 - np.array(range(min(len(tok_mbart_ls), 125)))
            id_sen_dict[id] = {}
            id_sen_dict[id]['score'] = 0
            id_sen_dict[id]['text'] = [ref_line]
            id_sen_score_dict[id] = [ref_line + " [Score: 0]"]
    # determine each sentence with specified number of noises
    sen_noise_dict, max_step = noise_planner(num_var, len(ref_lines), noise_planner_num)

    # load in mnli model for severity measures
    mnli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    mnli_model.eval()
    mnli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    m = nn.Softmax(dim=1)
    # batch_size_gen = 1
    # batch_size_xlm = 128
    # batch_size_mnli = 128
    print("Max Step: ", max_step)
    if log_file is not None:
        log_file.write(f"Max Step: {max_step} \n")
        log_file.flush()
    log_file.write(f"Number of sentences to corrupt is {len(id_sen_dict)} \n")
    for step in range(1, max_step + 1):
        mbart_add_seg_id_ls, mbart_add_start_ls, mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls, xlm_add_seg_id_ls, xlm_add_start_ls, \
        xlm_replace_seg_id_ls, xlm_replace_start_ls, swap_seg_id_ls, swap_start_ls, swap_end_ls, del_seg_id_ls, del_start_ls, del_len_ls, xlm_synonym_seg_id_ls, \
        xlm_synonym_start_ls, xlm_lemmatization_seg_id_ls, xlm_lemmatization_start_ls, step_noise_dict = noise_schedule(
            id_sen_dict, step, sen_noise_dict, cand_dict_arr, del_noise_lam, mask_noise_lam, word_tokens=word_tokens,
            use_new_operators=use_new_operators)
        # produce the text for generate functions
        mbart_ls, xlm_ls, swap_ls, delete_ls, synonym_ls, lemmatization_ls = [], [], [], [], [], []
        # construct mbart add dataset
        start_time = time.time()
        for id, start_index in zip(mbart_add_seg_id_ls, mbart_add_start_ls):
            mbart_ls.append(
                data_construct(id_sen_dict[id]['text'][-1], 1, mbart_tokenizer, start_index, 0, word_tokens=word_tokens,
                               pmi=pmi))
        # construct mbart replace dataset
        if log_file is not None:
            log_file.write(f"Done Mbart add {time.time() - start_time:.4f} seconds \n")
            log_file.flush()
        print(f"Done Mbart add {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        for id, start_index, replace_len in zip(mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls):
            mbart_ls.append(data_construct(id_sen_dict[id]['text'][-1], 2, mbart_tokenizer, start_index, replace_len,
                                           word_tokens=word_tokens, pmi=pmi))
        if log_file is not None:
            log_file.write(f"Done Mbart replace {time.time() - start_time:.4f} seconds \n")
            log_file.flush()
        print(f"Done Mbart replace {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        # construct xlm add dataset
        for id, start_index in zip(xlm_add_seg_id_ls, xlm_add_start_ls):
            xlm_ls.append(
                data_construct(id_sen_dict[id]['text'][-1], 3, xlm_tokenizer, start_index, 0, word_tokens=word_tokens,
                               pmi=pmi))
        if log_file is not None:
            log_file.write(f"Done add {time.time() - start_time:.4f} seconds \n")
            log_file.flush()
        print(f"Done add {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        # construct xlm replace dataset
        for id, start_index in zip(xlm_replace_seg_id_ls, xlm_replace_start_ls):
            xlm_ls.append(
                data_construct(id_sen_dict[id]['text'][-1], 4, xlm_tokenizer, start_index, 1, word_tokens=word_tokens,
                               pmi=pmi))
        if log_file is not None:
            log_file.write(f"Done replace {time.time() - start_time:.4f} seconds \n")
            log_file.flush()
        print(f"Done replace {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        # construct swap dataset
        for id, start_index, end_index in zip(swap_seg_id_ls, swap_start_ls, swap_end_ls):
            swap_ls.append(
                data_construct(id_sen_dict[id]['text'][-1], 5, xlm_tokenizer, start_index, end_index,
                               word_tokens=word_tokens, pmi=pmi))
        log_file.flush()
        if log_file is not None:
            log_file.write(f"Done swap {time.time() - start_time:.4f} seconds \n")
            log_file.flush()
        print(f"Done swap {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        # construct del dataset
        for id, start_index, del_len in zip(del_seg_id_ls, del_start_ls, del_len_ls):
            delete_ls.append(
                data_construct(id_sen_dict[id]['text'][-1], 6, xlm_tokenizer, start_index, del_len,
                               word_tokens=word_tokens, pmi=pmi))
        if log_file is not None:
            log_file.write(f"Done delete {time.time() - start_time:.4f} seconds \n")
            log_file.flush()
        print(f"Done delete {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        for id, start_index in zip(xlm_synonym_seg_id_ls, xlm_synonym_start_ls):
            synonym_ls.append(
                data_construct(id_sen_dict[id]['text'][-1], 7, xlm_tokenizer, start_index, 0, word_tokens=word_tokens,
                               pmi=pmi, analysis_logger=analysis_logger))
        if log_file is not None:
            log_file.write(f"Done synonyms {time.time() - start_time:.4f} seconds \n")
            log_file.flush()
        print(f"Done synonyms {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        for id, start_index in zip(xlm_lemmatization_seg_id_ls, xlm_lemmatization_start_ls):
            lemmatization_ls.append(
                data_construct(id_sen_dict[id]['text'][-1], 8, xlm_tokenizer, start_index, 0, word_tokens=word_tokens,
                               pmi=pmi))
        if log_file is not None:
            log_file.write(f"Done lemmatization {time.time() - start_time:.4f} seconds \n")
            log_file.flush()
        print(f"Done lemmatization {time.time() - start_time:.4f} seconds")
        if log_file is not None:
            log_file.write("All <mask>/non <mask> datasets are constructed for generation \n")
            log_file.flush()
        print("All <mask>/non <mask> datasets are constructed for generation")
        # sentence seg id with corresponding generated texts
        new_seg_ids, new_step_ls, step_score_ls = [], [], []

        new_seg_ids.extend(mbart_add_seg_id_ls)  # add in all seg ids for mbart add
        new_seg_ids.extend(mbart_replace_seg_id_ls)  # add in all seg ids for mbart replace
        # add all generated mbart add/replace texts into the new_step_ls
        mbart_model = mbart_model.to(device)
        if log_file is not None and step == 1:
            log_file.write(f"Number of sentences for mbart is {len(mbart_ls)} \n")
            log_file.flush()
        print(len(mbart_ls))
        start_time = time.time()
        for mbart_batch in tqdm(batchify(mbart_ls, mbart_batch_size)):
            # TODO: Figure out how to make it work
            mbart_texts = mbart_generation(mbart_batch, mbart_model, lang, mbart_tokenizer, device, word_tokens)
            new_step_ls.extend(mbart_texts)
        if log_file is not None and step == 1:
            log_file.write(f"Mbart time is {time.time() - start_time:.4f} \n")
            log_file.flush()
        mbart_model = mbart_model.to('cpu')
        torch.cuda.empty_cache()
        new_seg_ids.extend(xlm_add_seg_id_ls)  # add in all seg ids for xlm add
        new_seg_ids.extend(xlm_replace_seg_id_ls)  # add in all seg ids for xlm replace
        # add all generated xlm add/replace texts into the new_step_ls
        xlm_model = xlm_model.to(device)
        if log_file is not None:
            log_file.write(f"Number of sentences for xlm is {len(xlm_ls)} \n")
            log_file.flush()
        print(len(xlm_ls))
        start_time = time.time()
        for xlm_batch in tqdm(batchify(xlm_ls, xlm_batch_size)):
            xlm_texts = xlm_roberta_generate(xlm_batch, xlm_model, xlm_tokenizer, device, word_tokens)
            new_step_ls.extend(xlm_texts)
        if log_file is not None:
            log_file.write(f"xlm time is {time.time() - start_time:.4f} \n")
            log_file.flush()
        xlm_model = xlm_model.to('cpu')
        torch.cuda.empty_cache()
        new_seg_ids.extend(swap_seg_id_ls)  # add in all seg ids for swap
        new_step_ls.extend(swap_ls)
        new_seg_ids.extend(del_seg_id_ls)  # add in all seg ids for delete
        new_step_ls.extend(delete_ls)

        new_seg_ids.extend(xlm_synonym_seg_id_ls)
        new_step_ls.extend(synonym_ls)
        new_seg_ids.extend(xlm_lemmatization_seg_id_ls)
        new_step_ls.extend(lemmatization_ls)

        # update all cand dict arr for add/replace from xlm, swap and delete noises
        cand_dict_arr = add_update_cand_dict(cand_dict_arr, xlm_add_seg_id_ls, xlm_add_start_ls)
        cand_dict_arr = replace_update_cand_dict(cand_dict_arr, xlm_replace_seg_id_ls, xlm_replace_start_ls)
        cand_dict_arr = swap_update_cand_dict(cand_dict_arr, swap_seg_id_ls, swap_start_ls, swap_end_ls)
        cand_dict_arr = delete_update_cand_dict(cand_dict_arr, del_seg_id_ls, del_start_ls, del_len_ls)

        cand_dict_arr = replace_update_cand_dict(cand_dict_arr, xlm_synonym_seg_id_ls, xlm_synonym_start_ls)
        cand_dict_arr = replace_update_cand_dict(cand_dict_arr, xlm_lemmatization_seg_id_ls, xlm_lemmatization_start_ls)
        prev_step_ls = prev_ids_sens_extract(id_sen_dict, new_seg_ids)
        if step == 1:
            original = prev_step_ls.copy()
        if log_file is not None:
            log_file.write("Finish one step sentence generation! \n")
            log_file.flush()
        print("Finish one step sentence generation!")
        mnli_model = mnli_model.to(device)
        start_time = time.time()
        # use MNLI Roberta large model to determine the severities and scores
        if analysis_logger is not None:
            analysis_logger.write("Score analysis \n")
        for prev_batch, cur_batch, idx_batch in tqdm(zip(batchify(prev_step_ls, mnli_batch_size),
                                                         batchify(new_step_ls, mnli_batch_size),
                                                         batchify(new_seg_ids, mnli_batch_size))):
            if severity == 'original':
                temp_scores_ls = severity_measure(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device)
            elif severity == 'positive_correction':
                temp_scores_ls = severity_measure_with_positive_correction(mnli_model, mnli_tokenizer, m, prev_batch,
                                                                           cur_batch, device,
                                                                           ref_lines, idx_batch, lam=args.lam,
                                                                           analysis_logger=analysis_logger)

            elif severity == "continuous_scoring_geometric_mean":
                temp_scores_ls = severity_measure_with_continuous_scoring_geometric_mean(mnli_model, mnli_tokenizer, m, prev_batch,
                                                                          cur_batch, device, n=args.n,
                                                                          analysis_logger=analysis_logger)
            elif severity == "continuous_scoring_arithmetic_mean":
                temp_scores_ls = severity_measure_with_continuous_scoring_arithmetic_mean(mnli_model, mnli_tokenizer, m, prev_batch,
                                                                          cur_batch, device, n=args.n,
                                                                          analysis_logger=analysis_logger)
            elif severity == "constant_score":
                temp_scores_ls = np.ones(len(prev_batch)) * -5
            else:
                raise ValueError("Severity scoring not recognized!")
            step_score_ls.extend(temp_scores_ls)
        if log_file is not None:
            log_file.write("Finish one step MNLI! \n")
            log_file.write(f"Mnli time is {time.time() - start_time:.4f} \n")
            log_file.flush()
        print("Finish one step MNLI!")
        mnli_model = mnli_model.to('cpu')
        torch.cuda.empty_cache()
        # update all the sentences and scores in the prev dict
        for id, new_sen, score in zip(new_seg_ids, new_step_ls, step_score_ls):
            new_sen = " ".join(new_sen.split())
            if new_sen not in id_sen_dict[id]['text']:
                id_sen_dict[id]['text'].append(new_sen)
                id_sen_dict[id]['score'] += score
                id_sen_score_dict[id].append(new_sen + f" [Score: {score}, Info: {step_noise_dict[id]}]")
            else:
                print(id_sen_dict[id]['text'])
                print(new_sen)
        print("Finish one step")

    return id_sen_dict, id_sen_score_dict


def parser_args():
    """To make pmi or words token True add to configuration -pmi or -word_tokens"""
    args = argparse.ArgumentParser()
    args.add_argument('-num_var', type=int, default=5)
    args.add_argument('-lang', type=str, default='en_XX')
    args.add_argument('-ref', type=str, default='case_study_ref/wmt_train_fixed.txt')
    args.add_argument('-save_name_start', type=str, default='save_file_name')
    args.add_argument('-severity', type=str, default='original')
    args.add_argument('-word_tokens', default=False, action='store_true')
    args.add_argument('-pmi', default=False, action='store_true')
    args.add_argument('-mbart_batch_size', default=2, type=int)
    args.add_argument('-xlm_batch_size', default=8, type=int)
    args.add_argument('-mnli_batch_size', default=128, type=int)
    args.add_argument('-seed', default=12, type=int)
    args.add_argument('-noise_planner_num', default=1.5, type=float)
    args.add_argument('-del_noise_lam', default=1.5, type=float)
    args.add_argument('-mask_noise_lam', default=1.5, type=float)
    args.add_argument('-use_new_operators', default=False, action='store_true')
    args.add_argument('-save_dir', default="", type=str)
    args.add_argument('-logger', default="log.txt", type=str)
    args.add_argument('-lam', default=1, type=int)
    args.add_argument('-analysis_mode', default=False, action='store_true')
    args.add_argument('-analysis_save_name', default=None, type=str)
    args.add_argument('-analysis_save_dir',default="analysis_data")
    args.add_argument('-n', default=6.5788, type=float)
    return args.parse_args()


def main(args):
    """num_var: specifies number of different variants we create for each segment, lang: language code for model,
    src: source folder, ref: reference folder, save: file to save all the generated noises"""
    # load into reference file
    num_var = args.num_var
    lang = args.lang
    ref = args.ref
    print(ref)
    save_name_start = args.save_name_start
    save_dir = args.save_dir
    seed = args.seed
    random.seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = 'cpu'
    noise_planner_num = args.noise_planner_num
    del_noise_lam = args.del_noise_lam
    mask_noise_lam = args.mask_noise_lam
    save_name = save_name_start + f'_num_{noise_planner_num}_del_{del_noise_lam}_mask_{mask_noise_lam}_xlm_mbart.csv'
    dir_path = save_dir + '/' + save_name[:-4]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    csvfile = open(dir_path + '/' + save_name, 'w')
    csvwriter = csv.writer(csvfile)
    # fields = ['src', 'mt', 'ref', 'score']
    fields = ['mt', 'ref', 'score']
    csvwriter.writerow(fields)

    segFile = open(
        dir_path + '/' + f"{save_name_start}_zhen_num_{noise_planner_num}_del_{del_noise_lam}_mask_{mask_noise_lam}_xlm_mbart.tsv",
        'wt')
    tsv_writer = csv.writer(segFile, delimiter='\t')

    # for src_file, ref_file in zip(sorted(list(glob.glob(src + '/*'))), sorted(list(glob.glob(ref + '/*')))):

    # ref_file = 'case_study_ref/wmt_train.txt'
    ref_lines = open(ref, 'r').readlines()
    # src_lines = open(src_file, 'r').readlines()
    ref_lines = [" ".join(line[:-1].split()) for line in ref_lines]
    # src_lines = [" ".join(line[:-1].split()) for line in src_lines]
    if args.word_tokens:
        ref_lines = adjust_punctuation(ref_lines)
    logger = args.logger
    if logger is not None:
        logger_path = dir_path + '/' + logger
        log_file = open(logger_path, 'w')
    else:
        log_file = None
    if log_file is not None:
        log_file.write(f"Text Preprocessed to remove newline and Seed: {seed}\n")
        log_file.write(str(args) + '\n')
        log_file.flush()
    print(f"Text Preprocessed to remove newline and Seed: {seed}")

    start = time.time()
    try:
        id_sen_dict, id_sen_score_dict = text_score_generate(num_var, lang, ref_lines, noise_planner_num, del_noise_lam,
                                                             mask_noise_lam, device, args, log_file, dir_path)
        import pickle
        step_noise_dict_path = dir_path + f'/sentences_dict.pickle'
        with open(step_noise_dict_path, 'wb') as f:
            pickle.dump(id_sen_score_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        if log_file is not None:
            log_file.write(f"Total generated sentences for one subfile: {len(id_sen_dict)}\n")
            log_file.flush()

        print("Total generated sentences for one subfile: ", len(id_sen_dict))

        for key, value in id_sen_dict.items():
            seg_id = int(key.split('_')[0])
            noise_sen, score = value['text'][-1], value['score']  # the last processed noise sentence
            # csvwriter.writerow([src_lines[seg_id], noise_sen, ref_lines[seg_id], score])
            csvwriter.writerow([noise_sen, ref_lines[seg_id], score])

        for _, values in id_sen_score_dict.items():
            tsv_writer.writerow(values)
        if log_file is not None:
            log_file.write(f"Finished in {time.time() - start} seconds \n")
            log_file.write(f"{csvfile} Subfile outputs are saved in regression csv format! \n")
            log_file.flush()
        print(f"Finished in {time.time() - start} seconds")
        print(f"{csvfile} Subfile outputs are saved in regression csv format!")
    except Exception as Argument:
        if log_file is not None:
            log_file.write(f"{str(Argument)} \n")
            import traceback
            log_file.write(traceback.format_exc())
            log_file.flush()


if __name__ == "__main__":
    args = parser_args()
    main(args)
