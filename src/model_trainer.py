import argparse
from entities.dataset import Dataset
from extractors.tfidf_extractor import TfIdfExtractor
from extractors.scake_model import SCAKEModel
from evaluation import evaluate

DATASET = {
    "all": "./../data",
    "rt": "./../data/russia_today",
    "ng": "./../data/ng",
    "habr": "./../data/habrahabr",
    "cl": "./../data/cyberleninka"
}


def tfidf_extractor(dataset, ouput_path):
    extractor = TfIdfExtractor()
    predicted_kws = extractor.predict(dataset.get_texts()[:10], top_n=10)
    extractor.save_to_path(ouput_path)
    return predicted_kws


def scake_extractor(dataset, _output_path):
    extractor = SCAKEModel()
    return extractor.predict(dataset.get_tokens_and_ngrams()[:10], top_n=10)


MODELS = {
    "tfidf": tfidf_extractor,
    "scake": scake_extractor
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Keyword extractor')
    parser.add_argument("-d", "--dataset", dest="dataset", type=lambda x: DATASET.get(x), nargs="?", default="ng",
                        choices=DATASET.keys())
    parser.add_argument("-o", "--output", dest="output_path", type=str, default="model.pickle")
    parser.add_argument("-m", "--model", dest="model", type=lambda x: MODELS.get(x), default="tfidf", nargs="?",
                        choices=MODELS.keys())
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = Dataset(args.dataset, normalize=True)
    predicted_kws = args.model(dataset, args.output_path)
    true_kws = dataset.get_labels()[:10]
    evaluate(true_kws, predicted_kws)


if __name__ == '__main__':
    main()
