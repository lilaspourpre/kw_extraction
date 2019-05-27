from collections import Counter
import numpy as np

from extractors.model import Model


class GraphModel(Model):
    def __init__(self, texts):
        tokens = [token for text in texts for token in text.split()]
        self.word2id, self.id2word = self.build_dictionary(tokens)
        self.matrix = self.create_matrix(texts, self.word2id)

    def predict(self, dataset):
        pass

    def create_matrix(self, texts, word2id):
        columns = np.array([self.generate_column(word2id, Counter(tokens)) for tokens in texts])
        adjacency_matrix = np.matmul(columns, columns.transpose())
        print(np.array(adjacency_matrix).shape)
        exit()

    def generate_column(self, word2id, counter):
        return [counter.get(word2id[k], 0) for k in word2id.keys()]

    def build_dictionary(self, text_tokens):
        tokens = enumerate(set([token for tokens in text_tokens for token in tokens]))
        return {token: i for i, token in tokens}, {i: token for i, token in tokens}
