from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pandas as pd

PUNCT = punctuation + '«»—…“”*№–'
STOP_WORDS = set(stopwords.words('russian'))


def read_data(path):
    files = [os.path.join(common_path, filename) for common_path, _, filenames in os.walk(path)
             for filename in filenames]
    return pd.concat([pd.read_json(file, lines=True, encoding='utf-8') for file in files][:1],
                     axis=0, ignore_index=True)


def tokenize(text):
    tokens = word_tokenize(text)
    return tokens


def normalize_text(words, morph):
    words = [morph.parse(word)[0] for word in words if word and word not in STOP_WORDS]
    words = [word.inflect({'nomn'}) for word in words if
             word.tag.POS == 'NOUN' or word.tag.POS == "ADJS" or word.tag.POS == "ADJF"]
    words = [word.word for word in words if word is not None]
    return " ".join(words)



