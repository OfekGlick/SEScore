import random
import click
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
import math
from nltk import ngrams
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
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


def noise_sanity_check(cand_arr, num_noises, del_noise_lam=None, mask_noise_lam=None):
    # decide noise type upon function called, only sentences have one noise and step 1 can have MBart noises
    if num_noises == 1:
        noise_type = \
            random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=(0, 0, 1 / 4, 1 / 4, 1 / 4, 0, 0, 0),
                           k=1)[
                0]
    else:
        noise_type = random.choices([3, 4, 5, 6, 7, 8], weights=(1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 0), k=1)[0]

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


def noise_schedule(id_sen_dict, step, sen_noise_dict, cand_dict_arr, del_noise_lam, mask_noise_lam):
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
                                                                  mask_noise_lam)
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


def severity_measure_2_1(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device, original_sen, segs, lam=1,
                         delta=1):
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

    for prob_1, prob_2, pi_10, p0i_1, pi0, p0i in zip(softmax_result_1, softmax_result_2, softmax_result_org_prev_1,
                                                      softmax_result_org_prev_2, softmax_result_org_curr_1,
                                                      softmax_result_org_curr_2):
        ratio1 = pi0 / pi_10
        ratio2 = p0i / p0i_1
        cond1 = ratio1 > delta and ratio2 > delta
        cond2 = prob_1 > 0.9 and prob_2 > 0.9
        scores.append((-1 if cond2 > 0.9 else -5) + lam * (1 if cond1 else 0))
    return scores


def severity_measure_2_2(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device):
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
    n = 6.5788
    for prob_1, prob_2 in zip(softmax_result_1, softmax_result_2):
        scores.append(-(1 - (((prob_1 / (1 / prob_1)) * prob_2 / (1 / prob_2)).item()) ** n))
    return scores


# The reason we work this way (include the word in this calculation) is because we want all the word
# tokens to be included.Then we remove 1 in order to work with the code.
# If we worked with the word before, we don't know how many tokens the next word is.
def infer_token_location_index_mbart(text, tokenizer, index):
    words = text.split(' ')
    sub_sen = '</s> ' + " ".join(words[:index + 1])
    tok_text = \
        tokenizer(sub_sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)[
            'input_ids'][0]
    # End tokens of </s> and en_XX
    new_start_index = len(tok_text) - 1 - 2
    return new_start_index


def infer_ops_num_and_start_index_mbart(text, tokenizer, start_index, num_ops):
    token_start = infer_token_location_index_mbart(text, tokenizer, start_index)
    words = text.split(' ')
    sub_sen = '</s> ' + " ".join(words[:start_index + num_ops + 1])
    tok_text = \
        tokenizer(sub_sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)[
            'input_ids'][0]
    new_ops = len(tok_text) - token_start - 1 - 2
    return token_start, new_ops


def infer_token_location_index_xlm(text, tokenizer, index):
    words = text.split(' ')
    sub_sen = " ".join(words[:index + 1])
    tok_text = \
        tokenizer(sub_sen, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
                  padding=True)[
            'input_ids'][0]
    new_start_index = len(tok_text)
    return new_start_index


def infer_ops_num_and_start_index_xlm(text, tokenizer, start_index, num_ops):
    token_start = infer_token_location_index_xlm(text, tokenizer, start_index)
    words = text.split(' ')
    sub_sen = " ".join(words[:start_index + num_ops + 1])
    tok_text = \
        tokenizer(sub_sen, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
                  padding=True)[
            'input_ids'][0]
    new_ops = len(tok_text) - token_start
    return token_start, new_ops


def data_construct(text, noise_type, tokenizer, start_index, num_ops, words_token=True, pmi=True):
    if not words_token:
        start_index += 1  # to incorporate beginning token
    if noise_type == 1:
        if words_token:
            start_index = infer_token_location_index_mbart(text, tokenizer, start_index)
        sen = '</s> ' + text
        if pmi:
            sen = sen.replace("☘", " ")
        tok_text = \
            tokenizer(sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True,
                      padding=True)[
                'input_ids']
        input_ids = torch.cat(
            (tok_text[0][:start_index + 2], torch.LongTensor([tokenizer.mask_token_id]),
             tok_text[0][start_index + 2:]),
            dim=0)  # index shifts by 1 bc of </s>
    elif noise_type == 2:
        if words_token:
            start_index, num_ops = infer_ops_num_and_start_index_mbart(text, tokenizer, start_index, num_ops)
        sen = '</s> ' + text
        if pmi:
            sen = sen.replace("☘", " ")
        tok_text = \
            tokenizer(sen, add_special_tokens=True, return_tensors="pt", max_length=128, truncation=True, padding=True)[
                'input_ids']
        input_ids = torch.cat((tok_text[0][:start_index + 2], torch.LongTensor([tokenizer.mask_token_id]),
                               tok_text[0][start_index + 2 + num_ops:]), dim=0)  # index shifts by 1 bc of </s>
    elif noise_type == 3:
        if words_token:
            start_index = infer_token_location_index_xlm(text, tokenizer, start_index)
        if pmi:
            text = text.replace("☘", " ")
        tok_text = \
            tokenizer(text, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
                      padding=True)[
                'input_ids']
        input_ids = torch.cat(
            (tok_text[0][:start_index + 1], torch.LongTensor([tokenizer.mask_token_id]), tok_text[0][start_index + 1:]),
            dim=0)
    elif noise_type == 4:
        if words_token:
            start_index, num_ops = infer_ops_num_and_start_index_xlm(text, tokenizer, start_index, num_ops)
        if pmi:
            text = text.replace("☘", " ")
        tok_text = \
            tokenizer(text, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
                      padding=True)[
                'input_ids']
        input_ids = torch.cat((tok_text[0][:start_index + 1], torch.LongTensor([tokenizer.mask_token_id]),
                               tok_text[0][start_index + 1 + num_ops:]), dim=0)
    elif noise_type == 5:
        if words_token:
            start_index = infer_token_location_index_xlm(text, tokenizer, start_index)
            num_ops = infer_token_location_index_xlm(text, tokenizer, num_ops)
        end_index = num_ops
        if pmi:
            text = text.replace("☘", " ")
        tok_text = \
            tokenizer(text, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
                      padding=True)[
                'input_ids']
        input_ids = torch.cat((tok_text[0][:start_index + 1], torch.unsqueeze(tok_text[0][end_index + 1], 0),
                               tok_text[0][start_index + 2:end_index + 1],
                               torch.unsqueeze(tok_text[0][start_index + 1], 0), tok_text[0][end_index + 2:]), dim=0)
        return tokenizer.decode(input_ids, skip_special_tokens=True)

    elif noise_type == 6:
        if words_token:
            start_index, num_ops = infer_ops_num_and_start_index_xlm(text, tokenizer, start_index, num_ops)
        if pmi:
            text = text.replace("☘", " ")
        tok_text = \
            tokenizer(text, add_special_tokens=False, return_tensors="pt", max_length=128, truncation=True,
                      padding=True)[
                'input_ids']
        input_ids = torch.cat((tok_text[0][:start_index + 1], tok_text[0][start_index + 1 + num_ops:]), dim=0)
        return tokenizer.decode(input_ids, skip_special_tokens=True)

    elif noise_type == 7:
        if pmi:
            text = text.replace("☘", " ")
        tokens = text.split(" ")
        tokens[start_index] = select_random_synonym(tokens[start_index])
        text = " ".join(tokens)
        return text
    else:
        if pmi:
            text = text.replace("☘", " ")
        words = text.split(' ')
        words[start_index] = lemmatize(words[start_index])
        text = ' '.join(words)
        return text

    return tokenizer.decode(input_ids, skip_special_tokens=False)


def mbart_generation(batch_text, model, lang_code, tokenizer, device):
    with torch.no_grad():
        batch = tokenizer(batch_text, return_tensors="pt", max_length=128, truncation=True, padding=True)[
            'input_ids'].to(device)
        translated_tokens = model.generate(batch, decoder_start_token_id=tokenizer.lang_code_to_id[lang_code])
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translation


def xlm_roberta_generate(batch_text, model, xlm_tokenizer, device):
    with torch.no_grad():  # need to add special tokens bc previous steps didn't
        input_ids = \
            xlm_tokenizer(batch_text, add_special_tokens=True, return_tensors='pt', max_length=128, truncation=True,
                          padding=True)['input_ids'].to(device)
        logits = model(input_ids).logits

        for i, ele_input_ids in enumerate(input_ids):
            try:
                masked_index = (ele_input_ids == xlm_tokenizer.mask_token_id).nonzero().item()
            except ValueError:
                print(batch_text[i])
                print(ele_input_ids)
                print(xlm_tokenizer.mask_token_id)
            probs = logits[i, masked_index].softmax(dim=0)
            values, predictions = probs.topk(4)
            pred = random.choices(predictions, k=1)[0]
            input_ids[i][masked_index] = pred
        return xlm_tokenizer.batch_decode(input_ids, skip_special_tokens=True)


def text_score_generate(num_var, lang, ref_lines, noise_planner_num, del_noise_lam, mask_noise_lam, device,
                        severity='original', word_tokens=True, pmi=True):
    # load in XLM-Roberta model
    # xlm_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    # xlm_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-lbase").to(device)
    xlm_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    xlm_model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(device)
    xlm_model.eval()
    # load in MBart model and its tokenzier
    mbart_model = MBartForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path="facebook/mbart-large-cc25").to(device)
    mbart_tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang=lang)
    mbart_model.eval()
    # initialize cand_dict_arr, sen_noise_dict, id_sen_dict: key->seg_noise id, value->sentence list
    cand_dict_arr = {}
    id_sen_score_dict = {}
    id_sen_dict = {}  # id_sen_dict is a dict containing "score" and "text" fields, "text" field is a list which contains a history of all generated sentences
    pmi_order = parse_pmi(False)
    for line_index, ref_line in enumerate(ref_lines):
        for i in range(num_var):
            id = str(line_index) + '_' + str(i)
            if word_tokens:
                words = ref_line.split(' ')
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
                                gram_txt_replace = gram_txt.replace(" ", "☘")
                                ref_line = ref_line.replace(gram_txt, gram_txt_replace)
                words = ref_line.split(' ')
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load in mnli model for severity measures
    mnli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    mnli_model.eval()
    mnli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    m = nn.Softmax(dim=1)

    batch_size_gen = 16
    batch_size_xlm = 128
    batch_size_mnli = 128
    print("Max Step: ", max_step)
    for step in range(1, max_step + 1):
        mbart_add_seg_id_ls, mbart_add_start_ls, mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls, xlm_add_seg_id_ls, xlm_add_start_ls, \
        xlm_replace_seg_id_ls, xlm_replace_start_ls, swap_seg_id_ls, swap_start_ls, swap_end_ls, del_seg_id_ls, del_start_ls, del_len_ls, xlm_synonym_seg_id_ls, \
        xlm_synonym_start_ls, xlm_lemmatization_seg_id_ls, xlm_lemmatization_start_ls, step_noise_dict = noise_schedule(
            id_sen_dict, step, sen_noise_dict, cand_dict_arr, del_noise_lam, mask_noise_lam)
        # produce the text for generate functions
        mbart_ls, xlm_ls, swap_ls, delete_ls, synonym_ls, lemmatization_ls = [], [], [], [], [], []
        # construct mbart add dataset
        for id, start_index in zip(mbart_add_seg_id_ls, mbart_add_start_ls):
            mbart_ls.append(data_construct(id_sen_dict[id]['text'][-1], 1, mbart_tokenizer, start_index, 0))
        # construct mbart replace dataset
        for id, start_index, replace_len in zip(mbart_replace_seg_id_ls, mbart_replace_start_ls, mbart_replace_len_ls):
            mbart_ls.append(data_construct(id_sen_dict[id]['text'][-1], 2, mbart_tokenizer, start_index, replace_len))
        # construct xlm add daatset
        for id, start_index in zip(xlm_add_seg_id_ls, xlm_add_start_ls):
            xlm_ls.append(data_construct(id_sen_dict[id]['text'][-1], 3, xlm_tokenizer, start_index, 0))
        # construct xlm replace dataset
        for id, start_index in zip(xlm_replace_seg_id_ls, xlm_replace_start_ls):
            xlm_ls.append(data_construct(id_sen_dict[id]['text'][-1], 4, xlm_tokenizer, start_index, 1))
        # construct swap dataset
        for id, start_index, end_index in zip(swap_seg_id_ls, swap_start_ls, swap_end_ls):
            swap_ls.append(data_construct(id_sen_dict[id]['text'][-1], 5, xlm_tokenizer, start_index, end_index))
        # construct del dataset
        for id, start_index, del_len in zip(del_seg_id_ls, del_start_ls, del_len_ls):
            delete_ls.append(data_construct(id_sen_dict[id]['text'][-1], 6, xlm_tokenizer, start_index, del_len))
        for id, start_index in zip(xlm_synonym_seg_id_ls, xlm_synonym_start_ls):
            synonym_ls.append(data_construct(id_sen_dict[id]['text'][-1], 7, xlm_tokenizer, start_index, 0))
        for id, start_index in zip(xlm_lemmatization_seg_id_ls, xlm_lemmatization_start_ls):
            lemmatization_ls.append(data_construct(id_sen_dict[id]['text'][-1], 8, xlm_tokenizer, start_index, 0))
        print("All <mask>/non <mask> datasets are constructed for generation")
        # sentence seg id with corresponding generated texts
        new_seg_ids, new_step_ls, step_score_ls = [], [], []

        new_seg_ids.extend(mbart_add_seg_id_ls)  # add in all seg ids for mbart add
        new_seg_ids.extend(mbart_replace_seg_id_ls)  # add in all seg ids for mbart replace
        # add all generated mbart add/replace texts into the new_step_ls
        for mbart_batch in batchify(mbart_ls, batch_size_gen):
            mbart_texts = mbart_generation(mbart_batch, mbart_model, lang, mbart_tokenizer, device)
            new_step_ls.extend(mbart_texts)

        new_seg_ids.extend(xlm_add_seg_id_ls)  # add in all seg ids for xlm add
        new_seg_ids.extend(xlm_replace_seg_id_ls)  # add in all seg ids for xlm replace
        # add all generated xlm add/replace texts into the new_step_ls
        for xlm_batch in batchify(xlm_ls, batch_size_xlm):
            xlm_texts = xlm_roberta_generate(xlm_batch, xlm_model, xlm_tokenizer, device)
            new_step_ls.extend(xlm_texts)

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
        print("Finish one step sentence generation!")

        # use MNLI Roberta large model to determine the severities and scores
        for prev_batch, cur_batch, idx_batch in zip(batchify(prev_step_ls, batch_size_mnli),
                                                    batchify(new_step_ls, batch_size_mnli),
                                                    batchify(new_seg_ids, batch_size_mnli)):
            if severity == 'original':
                temp_scores_ls = severity_measure(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device)
            elif severity == 'type_1':
                temp_scores_ls = severity_measure_2_1(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device,
                                                      ref_lines, idx_batch)
            else:
                temp_scores_ls = severity_measure_2_2(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device)
            step_score_ls.extend(temp_scores_ls)

        print("Finish one step MNLI!")

        # update all the sentences and scores in the prev dict
        for id, new_sen, score in zip(new_seg_ids, new_step_ls, step_score_ls):
            new_sen = " ".join(new_sen.split())
            if new_sen not in id_sen_dict[id]['text']:
                id_sen_dict[id]['text'].append(new_sen)
                id_sen_dict[id]['score'] += score
                id_sen_score_dict[id].append(new_sen + f" [Score: {score}, Info: {step_noise_dict[id]}]")
            # else:
            #     print(id_sen_dict[id]['text'])
            #     print(new_sen)

        print("Finish one step")

    return id_sen_dict, id_sen_score_dict


@click.command()
@click.option('-num_var')
@click.option('-lang')
@click.option('-src')
@click.option('-ref')
@click.option('-save')
@click.option('-severity')
def main(num_var, lang, src, ref, save, severity, words_token=True):
    """num_var: specifies number of different variants we create for each segment, lang: language code for model,
    src: source folder, ref: reference folder, save: file to save all the generated noises"""
    # load into reference file
    random.seed(12)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    noise_planner_num, del_noise_lam, mask_noise_lam = 1.5, 1.5, 1.5
    save_name = save + f'_num_{noise_planner_num}_del_{del_noise_lam}_mask_{mask_noise_lam}_xlm_mbart.csv'
    csvfile = open(save_name, 'w')
    csvwriter = csv.writer(csvfile)
    #fields = ['src', 'mt', 'ref', 'score']
    fields = ['mt', 'ref', 'score']
    csvwriter.writerow(fields)

    segFile = open(f"{save}_zhen_num_{noise_planner_num}_del_{del_noise_lam}_mask_{mask_noise_lam}_xlm_mbart.tsv", 'wt')
    tsv_writer = csv.writer(segFile, delimiter='\t')

    #for src_file, ref_file in zip(sorted(list(glob.glob(src + '/*'))), sorted(list(glob.glob(ref + '/*')))):

    ref_file = 'case_study_ref/wmt_train.txt'
    ref_lines = open(ref_file, 'r').readlines()
    # src_lines = open(src_file, 'r').readlines()
    ref_lines = [" ".join(line[:-1].split()) for line in ref_lines]
    #src_lines = [" ".join(line[:-1].split()) for line in src_lines]

    print("Text Preprocessed to remove newline and Seed: 12")

    start = time.time()
    id_sen_dict, id_sen_score_dict = text_score_generate(int(num_var), lang, ref_lines, noise_planner_num,
                                                         del_noise_lam, mask_noise_lam, device, severity,
                                                         words_token)
    print("Total generated sentences for one subfile: ", len(id_sen_dict))

    for key, value in id_sen_dict.items():
        seg_id = int(key.split('_')[0])
        noise_sen, score = value['text'][-1], value['score']  # the last processed noise sentence
        #csvwriter.writerow([src_lines[seg_id], noise_sen, ref_lines[seg_id], score])
        csvwriter.writerow([ noise_sen, ref_lines[seg_id], score])

    for _, values in id_sen_score_dict.items():
        tsv_writer.writerow(values)

    print(f"Finished in {time.time() - start} seconds")
    print(f"{csvfile} Subfile outputs are saved in regression csv format!")


if __name__ == "__main__":
    main()
