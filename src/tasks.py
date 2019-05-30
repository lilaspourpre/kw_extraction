from celery import Celery
from extractors.tfidf_model import TFIDFModel
from extractors.scake_model import SCAKEModel

BROKER_URL = 'redis://127.0.0.1:6379/0'
app = Celery('tasks', broker=BROKER_URL, backend='redis://127.0.0.1:6379/0')


@app.task
def extract_kws_with_tfidf(text, top_n=10):
    predictor = TFIDFModel("./../models/model.pickle")
    return predictor.predict(text, top_n)


@app.task
def extract_kws_with_scake(tokens_with_ngrams, top_n=10):
    predictor = SCAKEModel()
    return predictor.predict(tokens_with_ngrams, top_n)