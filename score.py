import pickle
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, BertTokenizer
import torch.nn as nn
import numpy as np
import argparse
import os
import csv


def parser_args():
    """To make pmi or words token True add to configuration -pmi or -word_tokens"""
    args = argparse.ArgumentParser()
    args.add_argument('-num_var', type=int, default=1)
    args.add_argument('-ref', type=str, default='case_study_ref/wmt_train_small_fixed.txt')
    args.add_argument('-batch_size', default=160, type=int)
    args.add_argument('-save_dir', default="ablation_study_results", type=str)
    args.add_argument('-lam', default=1, type=int)
    args.add_argument('-analysis_mode', default=False, action='store_true')
    args.add_argument('-analysis_save_dir', default="analysis_data", type=str)
    args.add_argument('-n', default=6.5788, type=float)
    args.add_argument('-pickle_path', type=str)
    args.add_argument('-severity', type=str)
    args.add_argument('-save_name', type=str)
    return args.parse_args()


def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:min(i + batch_size, len(lst))]


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


def severity_measure_with_continuous_scoring_geometric_mean(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch,
                                                            device, n,
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
        bde = (prob_1.item() * prob_2.item()) ** 0.5
        score = bde ** n - 1
        if analysis_logger is not None:
            analysis_logger.write(f"The continuous score was {score} \n")
            analysis_logger.write(prev_batch[i] + '\n')
            analysis_logger.write(cur_batch[i] + '\n')
            analysis_logger.flush()
        scores.append(score)
    return scores


def severity_measure_with_continuous_scoring_arithmetic_mean(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch,
                                                             device, n,
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
        bde = (prob_1.item() + prob_2.item()) / 2
        score = bde ** n - 1
        if analysis_logger is not None:
            analysis_logger.write(f"The continuous score was {score} \n")
            analysis_logger.write(prev_batch[i] + '\n')
            analysis_logger.write(cur_batch[i] + '\n')
            analysis_logger.flush()
        scores.append(score)
    return scores


def score_corrupt_sentences(ref_lines, args):
    path_to_pickle_dict = args.pickle_path
    batch_size = args.batch_size
    severity = args.severity
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = "cpu"
    analysis_logger = None
    if args.analysis_mode:
        analysis_save_dir = args.analysis_save_dir + f"/{args.save_name[:-4]}" + '_scores'
        if not os.path.exists(analysis_save_dir):
            os.mkdir(analysis_save_dir)
        else:
            raise Exception("Error in analysis save path, it already exists")
        analysis_logger = open(analysis_save_dir + '/analysis_logger.txt', 'w')
    id_sen_dict = {}
    for line_index, ref_line in tqdm(enumerate(ref_lines)):
        for i in range(args.num_var):
            id = str(line_index) + '_' + str(i)
            id_sen_dict[id] = {}
            id_sen_dict[id]['score'] = 0
            id_sen_dict[id]['text'] = [ref_line]
    mnli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    mnli_model.eval()
    mnli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    m = nn.Softmax(dim=1)
    with open(path_to_pickle_dict, 'rb') as f:
        sentences_dict = pickle.load(f)
    for i in range(1, 6):
        prev = []
        curr = []
        ids = []
        step_score_ls = []
        for id, sentences in sentences_dict.items():
            if len(sentences) > i:
                prev.append(sentences[i - 1])
                curr.append(sentences[i])
                ids.append(id)
        for prev_batch, cur_batch, idx_batch in tqdm(zip(batchify(prev, batch_size),
                                                         batchify(curr, batch_size),
                                                         batchify(ids, batch_size))):
            if severity == 'original':
                temp_scores_ls = severity_measure(mnli_model, mnli_tokenizer, m, prev_batch, cur_batch, device)
            elif severity == 'positive_correction':
                temp_scores_ls = severity_measure_with_positive_correction(mnli_model, mnli_tokenizer, m, prev_batch,
                                                                           cur_batch, device,
                                                                           ref_lines, idx_batch, lam=args.lam,
                                                                           analysis_logger=analysis_logger)

            elif severity == "continuous_scoring_geometric_mean":
                temp_scores_ls = severity_measure_with_continuous_scoring_geometric_mean(mnli_model, mnli_tokenizer, m,
                                                                                         prev_batch,
                                                                                         cur_batch, device, n=args.n,
                                                                                         analysis_logger=analysis_logger)
            elif severity == "continuous_scoring_arithmetic_mean":
                temp_scores_ls = severity_measure_with_continuous_scoring_arithmetic_mean(mnli_model, mnli_tokenizer, m,
                                                                                          prev_batch,
                                                                                          cur_batch, device, n=args.n,
                                                                                          analysis_logger=analysis_logger)
            elif severity == "constant_score":
                temp_scores_ls = np.ones(len(prev_batch)) * -5
            else:
                raise ValueError("Severity scoring not recognized!")
            step_score_ls.extend(temp_scores_ls)
        for id, new_sen, score in zip(ids, curr, step_score_ls):
            new_sen = " ".join(new_sen.split())
            if new_sen not in id_sen_dict[id]['text']:
                id_sen_dict[id]['text'].append(new_sen)
                id_sen_dict[id]['score'] += score
            else:
                print(id_sen_dict[id]['text'])
                print(new_sen)
    return id_sen_dict


def save_scores():
    args = parser_args()
    print(args.ref)
    dir_path = args.save_dir + '/' + args.save_name[:-4]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    csvfile = open(dir_path + '/' + args.save_name, 'w')
    csvwriter = csv.writer(csvfile)
    ref_lines = open(args.ref, 'r').readlines()
    ref_lines = [" ".join(line[:-1].split()) for line in ref_lines]
    id_sen_dict = score_corrupt_sentences(ref_lines, args)
    for key, value in id_sen_dict.items():
        seg_id = int(key.split('_')[0])
        noise_sen, score = value['text'][-1], value['score']
        csvwriter.writerow([noise_sen, ref_lines[seg_id], score])


if __name__ == "__main__":
    save_scores()
