from pymorphy2 import MorphAnalyzer
from helpers import *


class Dataset:
    def __init__(self, dataset_path, normalize=False):
        self.normalize = normalize
        self.data = read_data(dataset_path)
        self.data['tokens'] = self.data['content'].apply(tokenize)
        if normalize:
            self.morph = MorphAnalyzer()
            self.data['content'] = self.data['tokens'].apply(normalize_text, morph=self.morph)

    def get_texts(self):
        return self.data["content"].tolist()

    def get_labels(self):
        return self.data['keywords'].tolist()

    def get_tokens(self):
        return self.data['tokens'].tolist()
