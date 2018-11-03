from src.data.utils import UNKNOWN_TOKEN
from src.config import Config

class FeaturesGenerator():
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

        self.word_to_idx = self.build_dictionary(Config.DATA_PATHS['words'])
        self.tag_to_idx = self.build_dictionary(Config.DATA_PATHS['tags'])
        self.char_to_idx = self.build_dictionary(Config.DATA_PATHS['chars'], start=1)
        self.char_to_idx['ZERO_PAD'] = 0

        self.idx_to_tag = {idx: tag for tag, idx in
                           self.tag_to_idx.items()}

        self.nwords = len(self.word_to_idx)
        self.nchars = len(self.char_to_idx)
        self.ntags = len(self.tag_to_idx)

    def get_char_ids(self, word):
        """get the array of character ids"""
        return [self.char_to_idx[c] for c in word if c in self.char_to_idx]

    def get_word_id(self, word):
        """get the word id"""
        processed_word = Config.word_processing(word)
        if processed_word not in self.word_to_idx:
            return self.word_to_idx[UNKNOWN_TOKEN]

        return self.word_to_idx[processed_word]

    def compute_features(self, words):
        """
        return (list of [char_ids], list of word_id)
        """
        return [self.get_char_ids(w) for w in words], [self.get_word_id(w) for w in words]

    def minibatches(self, dataset, batch_size):
        """ generator of (sentence, tags) tuples in IDX format
        """
        x_batch, y_batch = [], []
        for words, tags in dataset:            
            x = self.compute_features(words)
            y = [self.tag_to_idx[tag] for tag in tags]

            if len(x_batch) == batch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []

            x_batch.append(x)
            y_batch.append(y)

        if len(x_batch) != 0:
            yield x_batch, y_batch
