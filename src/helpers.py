from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import pandas as pd

PUNCT = punctuation + '«»—…“”*№–'
STOP_WORDS = set(stopwords.words('russian'))


def read_data(path):
    files = [os.path.join(common_path, filename) for common_path, _, filenames in os.walk(path)
             for filename in filenames]
    return pd.concat([pd.read_json(file, lines=True, encoding='utf-8') for file in files][:1],
                     axis=0, ignore_index=True)


def normalize_text(text, morph):
    sentences = sent_tokenize(text, "russian")
    words_in_sentences = []
    for sentence in sentences:
        words = [word.strip(PUNCT) for word in sentence.lower().split()]
        morph_parsed = [morph.parse(word)[0] for word in words if word and word not in STOP_WORDS]
        nominative_form = [word.inflect({'nomn'}) for word in morph_parsed if word.tag.POS in ('NOUN', "ADJS", "ADJF")]
        words = [form.word for form in nominative_form if form is not None]
        words_in_sentences.append(words)
    return words_in_sentences


def split_sentences(text):
    return [word_tokenize(sentence) for sentence in sent_tokenize(text)]