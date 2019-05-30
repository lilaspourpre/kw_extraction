from extractors.tfidf_model import TFIDFModel
from extractors.scake_model import SCAKEModel


def tfidf_extractor(dataset, ouput_path, top_n=10):
    extractor = TFIDFModel(data=dataset.get_texts())
    predicted_kws = extractor.predict(dataset.get_texts(), top_n=top_n)
    extractor.save_to_path(ouput_path)
    return predicted_kws


def scake_extractor(dataset, _output_path, top_n=10):
    extractor = SCAKEModel()
    return extractor.predict(dataset.get_tokens_and_ngrams(), top_n=top_n)

