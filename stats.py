import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from nltk import ngrams
import argparse
import os
import numpy as np

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


def parse_pmi(owt=False):
    filename = f"PMI/pmi-{'owt-' if owt else ''}wiki-bc_clean.txt"
    with open(filename) as f:
        data = f.read().split("\n")
        result = sorted([(len(row), row) for row in data], key=lambda x: x[0], reverse=True)
        result = set([unit for _, unit in result])
    return result


def count_pmi(ref_lines):
    occurrences = {2: {"counter": 0, "examples": dict()}, 3: {"counter": 0, "examples": dict()},
                   4: {"counter": 0, "examples": dict()}}
    with_pmi = set()
    pmi_order = parse_pmi(False)
    for line_index, ref_line in tqdm(enumerate(ref_lines)):
        words = ref_line.split()
        used_indecies = [0 for _ in range(len(words))]
        for n in range(4, 1, -1):
            ngrams_lst = ngrams(ref_line.split(), n)
            for k, gram in enumerate(ngrams_lst):
                gram_txt = " ".join(gram)
                flag = not any([used_indecies[j] for j in range(k, k + n)])
                if gram_txt in pmi_order and flag:
                    for j in range(k, k + n):
                        used_indecies[j] = 1
                    with_pmi.add(line_index)
                    gram_txt_replace = gram_txt.replace(" ", "♣")
                    occurrences[n]['counter'] += 1
                    if gram_txt_replace not in occurrences[n]['examples'].keys():
                        occurrences[n]['examples'][gram_txt_replace] = 0
                    occurrences[n]['examples'][gram_txt_replace] += 1
                    ref_line = ref_line.replace(gram_txt, gram_txt_replace)
    return occurrences, len(with_pmi) / len(ref_lines)


def parser_args():
    args = argparse.ArgumentParser()
    args.add_argument('-save_dir', type=str)
    args.add_argument('-data_path', type=str)
    return args.parse_args()


def pmi_analysis():
    args = parser_args()
    data_type = args.data_path.split('/')[-1][:-4]
    save_path = args.save_dir + '/' + data_type
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError("File exists, delete those results to repeat analysis")

    log_file = open(save_path + '/pmi_results.txt', 'w')
    if "test" in data_type:
        data = pd.read_csv(args.data_path)
        ref_lines = data['ref'].to_list()
    else:
        ref_lines = open(args.data_path, 'r').readlines()
        ref_lines = [" ".join(line[:-1].split()) for line in ref_lines]
    ref_lines = adjust_punctuation(ref_lines)
    results, pmi_percentage = count_pmi(ref_lines)
    for key, items in results.items():
        log_file.write(f"Number of words {key} \n")
        log_file.write(f"Number of occurrences {items['counter']} \n")
        log_file.write(f"Number of examples {len(items['examples'])}\n")
        log_file.write(f"The top 10 most common examples are: \n")
        examples = [(phrase, count) for phrase, count in items['examples'].items()]
        examples = sorted(examples, reverse=True, key=lambda x: x[1])
        for i in range(10):
            x = examples[i][0]
            x = x.replace("♣", " ")
            log_file.write(x + '\n')
        log_file.write("\n")
        log_file.flush()

def extract_corrections_examples():
    args = parser_args()
    ref_lines = open(args.data_path, 'r').readlines()
    originals = ref_lines[::3]
    prev_sentences = ref_lines[1::3]
    curr_sentences = ref_lines[2::3]
    for original,prev,curr in zip(originals,prev_sentences,curr_sentences):
        print(original)
        print(prev)
        print(curr)

def extract_new_operators_statistics():
    args = parser_args()
    ref_lines = open(args.data_path, 'r').readlines()
    synonym_lines = [line for line in ref_lines if "Synonym_" in line]
    lemmatization_lines = [line for line in ref_lines if "Lemmatization_" in line]
    synonym_success_percentage = len([line for line in synonym_lines if "succeeded"])/len(synonym_lines)
    print(f"Synonym success percentage is {synonym_success_percentage*100:.4f}%")
    lemmatization_success_percentage = len([line for line in lemmatization_lines if "succeeded"])/len(lemmatization_lines)
    print(f"Lemmatization success percentage is {lemmatization_success_percentage*100:.4f}%")



def extract_continuous_score_distribution(eps = 0.1):
    args = parser_args()
    ref_lines = open(args.data_path, 'r').readlines()
    ref_lines = [line[:-1] for line in ref_lines if "The continuous score" in line]
    scores = [float(line.split()[-1]) for line in ref_lines]
    plt.hist(scores,bins=10,weights= np.ones_like(scores) / len(scores))
    plt.show()
    scores_eps_from_borders = [score for score in scores if ((score >-1+eps) and (score < -eps))]
    print(f"Uncertain scores percentage {len(scores_eps_from_borders)/len(scores) *100:.4f}%")
    plt.hist(scores_eps_from_borders,bins=10,range=(0,1),weights= np.ones_like(scores) / len(scores))
    plt.show()

if __name__ == "__main__":
    pmi_analysis()
