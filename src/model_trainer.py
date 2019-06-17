import argparse
from entities.dataset import Dataset
from kw_extractors import *
from evaluation import evaluate

DATASET = {
    "all": "./../data",
    "rt": "./../data/russia_today",
    "ng": "./../data/ng",
    "habr": "./../data/habrahabr",
    "cl": "./../data/cyberleninka"
}

MODELS = {
    "tfidf": tfidf_extractor,
    "scake": scake_extractor,
    "lstm": lstm_extractor
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Keyword extractor')
    parser.add_argument("-d", "--dataset", dest="dataset", type=lambda x: DATASET.get(x), nargs="?", default="ng",
                        choices=DATASET.keys())
    parser.add_argument("-o", "--output", dest="output_path", type=str, default="model.pickle")
    parser.add_argument("-m", "--model", dest="model", type=lambda x: MODELS.get(x), default="scake", nargs="?",
                        choices=MODELS.keys())
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = Dataset(args.dataset, normalize=True)
    predicted_kws = args.model(dataset, args.output_path)
    true_kws = dataset.get_token_keywords()
    evaluate(true_kws, predicted_kws)


if __name__ == '__main__':
    main()
