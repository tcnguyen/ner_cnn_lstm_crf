from src.data.conlldataset import CoNLLDataset
from src.model.cnn_lstm_crf_model import CharCNNLSTMCRFModel
from src.model.cnn_lstm_crf_elmo_model import CharCNNLSTMCRFElmoModel
from src.config import Config


def main():
    if Config.word_embeddings == 'glove':
        model = CharCNNLSTMCRFModel()
    elif Config.word_embeddings == 'elmo':
        model = CharCNNLSTMCRFElmoModel()

    model.build()
    # model.restore_session(Config.dir_model) # optional, restore weights

    # create datasets
    valid = CoNLLDataset(Config.DATA_PATHS['valid'])
    train = CoNLLDataset(Config.DATA_PATHS['train'])

    # train model
    model.train(train, valid)


if __name__ == "__main__":
    main()
