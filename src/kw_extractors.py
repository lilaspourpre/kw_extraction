import tensorflow as tf
from gensim.models.fasttext import FastText

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


def lstm_extractor(dataframe, output_path, _top_n=10, fasttext_path='../models/cc.ru.300.bin'):
    output_size = 1
    hidden_size = 512
    shapes = ([None, 300], [], [None, output_size])
    dataset = tf.data.Dataset.from_generator(generator=DatasetGenerator(dataframe, fasttext_path).get_generator,
                                             output_types=(tf.float32, tf.int32, tf.float32)) \
        .padded_batch(16, padded_shapes=shapes, drop_remainder=True)

    train_iterator = dataset.make_initializable_iterator()
    iterable_tensors = train_iterator.get_next()
    extractor = LSTMModel(iterable_tensors, hidden_size=hidden_size, output_size=output_size)
    extractor.train(train_iterator, output_path)


class DatasetGenerator:
    def __init__(self, dataset, model_path):
        self.dataset = dataset
        self.model = FastText.load_fasttext_format(model_path)

    def get_generator(self):
        for text_tokens, token_labels in zip(self.dataset.get_tokens(), self.dataset.get_labels()):
            yield [self.get_vector(token) for token in text_tokens], len(text_tokens), [[i] for i in token_labels]

    def get_vector(self, token):
        return self.model.wv[token]
