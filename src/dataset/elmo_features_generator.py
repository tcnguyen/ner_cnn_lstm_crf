import random

from src.dataset.word_processing import UNKNOWN_TOKEN
from src.config import Config


class ElmoFeaturesGenerator():
    def __init__(self):
        self.initilize()

    def build_dictionary(self, file_path, start=0):
        """ Create dictionary from file which contains a word per line
        """
        with open(file_path) as f:
            words = f.readlines()
            words = [w.strip() for w in words]
            return dict({w: i for i, w in enumerate(words, start)})

    def initilize(self):
        """Loads vocabulary, processing functions and embeddings
        """
        self.tag_to_idx = self.build_dictionary(Config.DATA_PATHS['tags'])
        self.char_to_idx = self.build_dictionary(
            Config.DATA_PATHS['chars'], start=1)
        self.char_to_idx['ZERO_PAD'] = 0

        self.idx_to_tag = {idx: tag for tag, idx in
                           self.tag_to_idx.items()}

        self.nchars = len(self.char_to_idx)
        self.ntags = len(self.tag_to_idx)

    def get_char_ids(self, word):
        """get the array of character ids"""
        return [self.char_to_idx[c] for c in word if c in self.char_to_idx]

    def compute_features(self, words):
        """
        return (list of [char_ids], list of words)

        example: 
        words = ["I", "love", "Paris"]

          ------------------ char_ids ---------------   ------ words ------      
        ( [[27], [5, 17, 56, 79], [35, 2, 36, 52, 15]], ["I", "love", "Paris"])
        """
        return [self.get_char_ids(w) for w in words], words

    def minibatches(self, dataset, batch_size, shuffle=True):
        """ generator of (sentence, tags) tuples in IDX format

        TO IMPROVE: use tensorflow dataset
        """

        X = []  # load all data to shuffle
        Y = []  # load all data to shuffle

        for words, tags in dataset:
            x = self.compute_features(words)
            y = [self.tag_to_idx[tag] for tag in tags]

            X.append(x)
            Y.append(y)

        if shuffle:
            tmp = list(zip(X, Y))
            random.shuffle(tmp)
            X, Y = zip(*tmp)

        N = len(X)

        for i in range(0, N, batch_size):
            x_batch, y_batch = X[i:i+batch_size], Y[i:i+batch_size]
            yield x_batch, y_batch
