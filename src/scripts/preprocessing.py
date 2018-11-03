import pandas as pd
import numpy as np

from src.data.conlldataset import CoNLLDataset
from src.data.word_processing import UNKNOWN_TOKEN, PADDING_TOKEN
from src.config import Config


def get_all_words_and_tags():
    """
    Get all (processed) words and tags from all the train, dev, and test dataset
    """
    all_words = set()
    all_tags = set()

    for data_type in ['train', 'valid', 'test']:
        dataset = CoNLLDataset(Config.DATA_PATHS[data_type])
        for words, tags in dataset:
            # transform words
            processed_words = [Config.word_processing(w) for w in words]
            all_words.update(processed_words)
            all_tags.update(tags)

    return all_words, all_tags


def get_all_characters():
    all_chars = set()
    
    dataset = CoNLLDataset(Config.DATA_PATHS['train'])
    for words, _ in dataset:
        for word in words:
            all_chars.update(set([c for c in word]))

    return all_chars


def get_vocabulary_and_word_embeddings(data_words, embedding_file_path):
    """
    Extract all the words in "data_words" that have word embedding
    """
    df = pd.read_csv(embedding_file_path, sep=" ", quoting=3, header=None, index_col=0)
    embedded_words = set(df.index)
    print("pretrained embeddings vocabulary size: ", len(embedded_words))
    
    words = list(data_words & embedded_words)
    df = df.loc[words]

    special_tokens = [PADDING_TOKEN, UNKNOWN_TOKEN]
    vocabulary = special_tokens + list(df.index)
    word_embeddings = np.zeros((len(vocabulary), Config.EMBEDDING_SIZE))

    # padding and unknown token will have 0 embedding values
    # otherwise take pretrained embedding values
    word_embeddings[len(special_tokens):] = df.values

    return vocabulary, word_embeddings


def write_list(data, filename):
    with open(filename, "w") as f:
        for i, d in enumerate(data, 1):
            f.write(d)
            if i < len(data):
                f.write("\n")


if __name__ == '__main__':
    print("get all words and tags from data")
    all_words, all_tags = get_all_words_and_tags()
    print("found {} words, {} tags".format(len(all_words), len(all_tags)))

    print("get all characters from training data")
    all_chars = get_all_characters()
    write_list(sorted(list(all_chars)), Config.DATA_PATHS['chars'])

    print("build vocabulary and word embeddings")
    vocabulary, word_embeddings = get_vocabulary_and_word_embeddings(
        all_words, embedding_file_path=Config.EMBEDDING_PATH)

    print("write vocabulary and word embeddings")
    write_list(vocabulary, Config.DATA_PATHS['words'])
    write_list(all_tags, Config.DATA_PATHS['tags'])
    np.savez_compressed(Config.DATA_PATHS['word_embeddings'],
                        embeddings=word_embeddings)
