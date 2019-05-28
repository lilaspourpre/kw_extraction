from celery import Celery
from extractors.predictor import Predictor

BROKER_URL = 'redis://127.0.0.1:6379/0'
app = Celery('tasks', broker=BROKER_URL, backend='redis://127.0.0.1:6379/0')


@app.task
def extract_kws_from_text(text, top_n=10):
    predictor = Predictor("./../models/model.pickle")
    return predictor.predict(text, top_n)
