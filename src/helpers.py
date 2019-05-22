import nltk
from string import punctuation
from nltk.corpus import stopwords

PUNCT = punctuation + '«»—…“”*№–'
STOP_WORDS = set(stopwords.words('russian'))


def normalize_text(text):
    tokenized_text = [word.strip(PUNCT) for word in nltk.tokenize.word_tokenize(text) if word not in STOP_WORDS]
    return " ".join(tokenized_text)
