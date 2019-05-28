from pymorphy2 import MorphAnalyzer
from nltk.util import ngrams

from helpers import *


class Dataset:
    def __init__(self, dataset_path, normalize=False):
        self.normalize = normalize
        self.data = read_data(dataset_path)
        if normalize:
            self.morph = MorphAnalyzer()
            self.data['sentences'] = self.data['content'].apply(normalize_text, morph=self.morph)
        else:
            self.data['sentences'] = self.data['content'].apply(split_sentences)

    def get_raw_texts(self):
        return self.data["content"].tolist()

    def get_texts(self):
        normalized_texts = []
        for text in self.data["sentences"].tolist():
            normalized_texts.append(" ".join([" ".join(sentence) for sentence in text]))
        return normalized_texts

    def get_labels(self):
        return self.data['keywords'].tolist()

    def get_sentences(self):
        return self.data['sentences'].tolist()

    def get_ngrams(self, tokens, n=2):
        return [" ".join(i) for i in list(ngrams(tokens, n))]

    def get_tokens_and_ngrams(self, n=2):
        texts = self.data['sentences']
        updated_texts = []
        for text in texts:
            updated_texts.append([sentence + self.get_ngrams(sentence, n) for sentence in text])
        return updated_texts
