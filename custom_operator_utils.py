import random
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import spacy
import nltk


def parse_pmi(owt=False):
    filename = f"PMI/pmi-{'owt-' if owt else ''}wiki-bc_clean.txt"
    with open(filename) as f:
        data = f.read().split("\n")
        result = sorted([(len(row), row) for row in data], key=lambda x: x[0], reverse=True)
        result = set([unit for _, unit in result])
    return result


def select_random_synonym(word):
    if "♣" in word:
        words = word.split("♣")
        new_word = []
        for w in words:
            new_word.append(select_random_synonym(w))
        return " ".join(new_word)
    synonyms = set()
    synonyms.add(word)
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().split('_')
            if len(synonym) == 1:
                synonyms.add(synonym[0])
    return random.sample(synonyms, 1)[0]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize(word):
    if "♣" in word:
        words = word.split("♣")
        new_word = []
        for w in words:
            new_word.append(lemmatize(w))
        return " ".join(new_word)
    wml = WordNetLemmatizer()
    pos = get_wordnet_pos(pos_tag([word])[0][1])
    if pos is None:
        return word
    return wml.lemmatize(word, pos=pos)


# def correct_punctuation(sentence):
#     no_spaces = ["&","/","@","-","'"]
#     left_side_space = ["#","$","(","[","}"]
#     right_side_space = ["!","%",",",".",":","?",")","]","}"]
#     spaces_both_sides = ["*","+","<","=",">","|"]
#     depends = ['"']
#     new_sen = []
#     skip = False
#     depends_open = False
#     for i in range(len(sentence)):
#         if skip:
#             skip = False
#             continue
#         if sentence[i] in two_sides:
#             if sentence[i+1] == ' ':
#                 skip = True
#                 if new_sen[-1] == ' ':
#                     new_sen = new_sen.pop()
#         elif sentence[i] in left_side:
#             if new_sen[-1] != ' ':
#                 new_sen.append(' ')
#             if sentence[i + 1] == ' ':
#                 skip = True
#         elif sentence[i] in right_side:
#             if sentence[i + 1] != ' ':
#                 new_sen.append(sentence[i])
#                 new_sen.append(' ')
#                 continue
#         elif sentence[i] in depends:
#             if depends_open:
#                 if sentence[i + 1] != '':
#                     pass
