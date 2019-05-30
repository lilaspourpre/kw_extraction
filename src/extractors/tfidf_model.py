from sklearn.feature_extraction.text import TfidfVectorizer
from extractors.model import Model
import pickle
import os


class TFIDFModel(Model):
    instance = None

    class __TFIDFLoadModel:
        def __init__(self, model_path):
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
                self.id2word = {i: word for i, word in enumerate(self.model.get_feature_names())}

    class __TFIDFModel:
        def __init__(self, data):
            self.model = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            self.model.fit(data)
            self.id2word = {i: word for i, word in enumerate(self.model.get_feature_names())}

    def __init__(self, model_path=None, data=None):
        if not TFIDFModel.instance:
            if model_path:
                TFIDFModel.instance = TFIDFModel.__TFIDFLoadModel(model_path)
            elif data:
                TFIDFModel.instance = TFIDFModel.__TFIDFModel(data)
            else:
                raise Exception("tfidf model cannot be initiated")

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def predict(self, texts, top_n=10):
        top_vectors = self.model.transform(texts).toarray().argsort()
        return [[self.id2word[w] for w in top] for top in top_vectors[:, :-top_n-1:-1]]

    def save_to_path(self, path):
        with open(os.path.join("./../models/", path), 'wb') as file_to_write:
            pickle.dump(self.model, file_to_write)
