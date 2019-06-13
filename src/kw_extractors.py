import tensorflow as tf
import numpy as np

from extractors.tfidf_model import TFIDFModel
from extractors.scake_model import SCAKEModel
from extractors.lstm_model import LSTMModel


def tfidf_extractor(dataset, ouput_path, top_n=10):
    extractor = TFIDFModel(data=dataset.get_texts())
    predicted_kws = extractor.predict(dataset.get_texts(), top_n=top_n)
    extractor.save_to_path(ouput_path)
    return predicted_kws


def scake_extractor(dataset, _output_path, top_n=10):
    extractor = SCAKEModel()
    return extractor.predict(dataset.get_tokens_and_ngrams(), top_n=top_n)


def lstm_extractor(dataset, output_path, top_n=10):
    def _generator():
        for text_tokens, token_labels in zip(dataset.get_tokens(), dataset.get_labels()):
            yield np.zeros((len(text_tokens), 100)), len(text_tokens), np.zeros((len(token_labels),2))
    shapes = ((100,), (1), (2,))
    dataset = tf.data.Dataset.from_generator(generator=_generator,
                                             output_types=(tf.float32, tf.int32, tf.float32),
                                             output_shapes=shapes).batch(16)
    dataset = dataset.make_initializable_iterator()
    extractor = LSTMModel(512)
    extractor.train(dataset)
    #extractor.predict()