import random
import nltk

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize


def select_random_synonym(word):
    synonyms = set()
    synonyms.add(word)
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.add(l.name())
    return random.sample(synonyms, 1)[0]


def lemmatize(word):
    wml = WordNetLemmatizer()
    tag = pos_tag(word)[0][1][0].lower()
    tag = tag if tag in ['a', 'r', 'n', 'v'] else None
    if tag is None:
        return word
    return wml.lemmatize(word, pos=tag)


# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# print(select_random_synonym("sdfsdfsfsd"))
# print(lemmatize("understands"))
# sentence = "My secretary is the only person who truly understands my stamp collecting obsession"
# new_sentence = []
# for index in range(len(sentence.split(" "))):
#     new_sentence.append(lemmatize(sentence, index))
# print(sentence)
# print(" ".join(new_sentence))