import os

from src.data.word_processing import transform_word


class Config:
    
    
    DATA_ROOT = './data/test/' # USE THIS './data/test/'   TO TEST THE CODE
    DATA_PATHS = {
        # original data
        'train': DATA_ROOT + 'train.txt',
        'valid': DATA_ROOT + 'valid.txt',
        'test':  DATA_ROOT + 'test.txt',

        # computed data 
        'words': DATA_ROOT + 'words.txt',
        'tags':  DATA_ROOT + 'tags.txt',
        'chars': DATA_ROOT + 'chars.txt',
        'word_embeddings': DATA_ROOT + "word_embeddings.npz"
    }

    EMBEDDING_PATH = '../../word_embeddings/glove.6B.300d.txt'
    EMBEDDING_SIZE = 300

    # general config
    dir_output = "outputs/train/"
    dir_model = dir_output + "model.weights/"

    # embeddings
    dim_word = EMBEDDING_SIZE
    dim_char = 50

    # word processing
    word_processing = transform_word

    # training
    train_embeddings = False
    nepochs = 20
    dropout = 0.5
    batch_size = 16
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 3

    # model hyperparameters
    conv_filter_size = 3
    conv_filters = 32
    hidden_size_lstm = 300  # lstm on word embeddings


# directory for training outputs
if not os.path.exists(Config.dir_output):
    os.makedirs(Config.dir_output)