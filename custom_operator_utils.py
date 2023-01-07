import random
import nltk

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import spacy
def select_random_synonym(word):
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
    # load_model = spacy.load("en_core_web_sm")
    # doc = load_model(text)
    # new_text = " ".join([token.lemma_ for token in doc])
    wml = WordNetLemmatizer()
    pos = get_wordnet_pos(pos_tag([word])[0][1])
    if pos is None:
        return word
    return wml.lemmatize(word, pos=pos)
mapping = {}

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