from pymorphy2 import MorphAnalyzer

from helpers import *


class Dataset:
    def __init__(self, dataset_path, normalize=False):
        self.normalize = normalize
        self.data = read_data(dataset_path)
        self.set_sentences(normalize)
        self.set_labels()

    def set_sentences(self, normalize):
        if normalize:
            self.data['sentences'] = self.data['content'].apply(normalize_text, morph=MorphAnalyzer())
        else:
            self.data['sentences'] = self.data['content'].apply(split_sentences)

    def set_labels(self):
        def compute_labels(row):
            set_kws = set([i for j in row['keywords'] for i in j.split()])
            sentences = row['sentences']
            return [1 if token in set_kws else 0 for sentence in sentences for token in sentence]
        self.data['labels'] = self.data.apply(compute_labels, axis=1)

    def get_raw_texts(self):
        return self.data["content"].tolist()

    def get_texts(self):
        return join_to_text(self.data["sentences"].tolist())

    def get_keywords(self):
        return self.data['keywords'].tolist()

    def get_token_keywords(self):
        return self.data['keywords'].apply(lambda keywords: [kw for kw_phrase in keywords for kw in kw_phrase.split()])

    def get_labels(self):
        return self.data['labels'].tolist()

    def get_tokens(self):
        return [[token for sentence in text for token in sentence] for text in self.data['sentences'].tolist()]

    def get_sentences(self):
        return self.data['sentences'].tolist()

    def get_tokens_and_ngrams(self, n=3):
        texts = self.data['sentences']
        return add_ngrams(texts, n=n)

    def get_shape(self):
        return self.data.shape[0]
