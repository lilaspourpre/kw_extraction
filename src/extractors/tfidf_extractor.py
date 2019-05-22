from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle


class TfIdfExtractor:
    def __init__(self, texts):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.tfidf.fit(texts)
        self.id2word = {i: word for i, word in enumerate(self.tfidf.get_feature_names())}

    def predict(self, dataset):
        texts_vectors = self.tfidf.transform(dataset)
        return [[self.id2word[w] for w in top] for top in texts_vectors.toarray().argsort()[:, :-11:-1]]

    def save_to_path(self, path):
        with open(os.path.join("./../models/", path), 'wb') as file_to_write:
            pickle.dump(self.tfidf, file_to_write)
        # load the content
        # tfidf = pickle.load(open("x_result.pkl", "rb"))