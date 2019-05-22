import argparse
from entities.dataset import Dataset
from extractors.tfidf_extractor import TfIdfExtractor
from evaluation import evaluate

DATASET = {
    "all": "./../data",
    "rt": "./../data/russia_today",
    "ng": "./../data/ng",
    "habr": "./../data/habrahabr",
    "cl": "./../data/cyberleninka"
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Keyword extractor')
    parser.add_argument("-d", "--dataset", dest="dataset", type=lambda x: DATASET.get(x), nargs="?", default="ng",
                        choices=DATASET.keys())
    return parser.parse_args()


def evaluate_extractor(extractor, test_data):
    true_kws = test_data.get_labels()
    predicted_kws = extractor.predict(test_data.get_texts())
    evaluate(true_kws, predicted_kws)


def main():
    args = parse_arguments()
    dataset = Dataset(args.dataset, normalize=False)

    extractor = TfIdfExtractor(dataset.get_texts())
    evaluate_extractor(extractor, dataset)
    extractor.save_to_path("tfidf.pickle")


if __name__ == '__main__':
    main()
