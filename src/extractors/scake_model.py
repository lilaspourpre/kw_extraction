from collections import Counter
import numpy as np
import networkx as nx

from extractors.model import Model


class SCAKEModel(Model):
    def __init__(self):
        pass

    def predict(self, dataset, top_n):
        result_kws = []
        for text in dataset:
            result_kws.append(self.extract_keywords(text, top_n))
        return result_kws

    def extract_keywords(self, tokens_and_ngrams, top_n):
        matrix, id2word = self.create_matrix(tokens_and_ngrams)
        G = nx.from_numpy_array(matrix)
        node2measure = dict(nx.pagerank(G))
        return [id2word[index] for index, measure in sorted(node2measure.items(), key=lambda x: -x[1])[:top_n]]

    def create_matrix(self, tokens_and_ngrams):
        word2id, id2word = self.build_dictionary(tokens_and_ngrams)
        columns = np.array([self.generate_column(word2id, items) for items in tokens_and_ngrams])
        adjacency_matrix = np.matmul(columns.transpose(), columns)
        for i in range(adjacency_matrix.shape[0]):
            adjacency_matrix[i][i] = 0
        return adjacency_matrix, id2word

    def generate_column(self, word2id, tokens):
        counter = Counter(tokens)
        column = [counter.get(k, 0) for k in word2id.keys()]
        return column

    def build_dictionary(self, text):
        tokens_and_ngrams_set = set([item for sentence in text for item in sentence])
        return {token: i for i, token in enumerate(tokens_and_ngrams_set)}, \
               {i: token for i, token in enumerate(tokens_and_ngrams_set)}

    def save_to_path(self, path):
        raise NotImplementedError

