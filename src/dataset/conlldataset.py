class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    """

    def __init__(self, file_path):
        """
        file_path: path to the file
        """
        self.file_path = file_path
        self.length = None

    def __iter__(self):
        with open(self.file_path) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split()
                    word, tag = ls[0], ls[3]
                    words.append(word)
                    tags.append(tag)

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = len([_ for _ in self])
        return self.length
