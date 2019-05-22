import os
import pandas as pd

from helpers import *


class Dataset:
    def __init__(self, dataset_path, normalize=False):
        files = [os.path.join(common_path, filename) for common_path, _, filenames in os.walk(dataset_path)
                 for filename in filenames]
        self.data = pd.concat([pd.read_json(file, lines=True, encoding='utf-8') for file in files][:1],
                              axis=0, ignore_index=True)
        self.data['normalized_content'] = self.data['content'].apply(
            normalize_text if normalize else lambda x: x)

    def get_texts(self):
        return self.data['normalized_content'].to_list()

    def get_labels(self):
        return self.data['keywords'].to_list()
