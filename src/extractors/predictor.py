import pickle


class Predictor:
    class __Predictor:
        def __init__(self, model_path):
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
                self.id2word = {i: word for i, word in enumerate(self.model.get_feature_names())}
    instance = None

    def __init__(self, model_path):
        if not Predictor.instance:
            Predictor.instance = Predictor.__Predictor(model_path)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def predict(self, text, top_n=10):
        top_vectors = self.model.transform([text]).toarray()[0].argsort()
        return [self.id2word[word] for word in top_vectors[:-top_n-1:-1]]
