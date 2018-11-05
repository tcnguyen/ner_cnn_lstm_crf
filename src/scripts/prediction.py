from src.dataset.conlldataset import CoNLLDataset
from src.model.cnn_lstm_crf_model import CharCNNLSTMCRFModel
from src.model.cnn_lstm_crf_elmo_model import CharCNNLSTMCRFElmoModel
from src.config import Config


def main():
    # build model
    if Config.word_embeddings == 'glove':
        model = CharCNNLSTMCRFModel()
    elif Config.word_embeddings == 'elmo':
        model = CharCNNLSTMCRFElmoModel()

    model.build()
    model.restore_session(Config.dir_model)

    # create dataset
    test = CoNLLDataset(Config.DATA_PATHS['test'])

    # evaluate and interact
    with open('test.preds.txt', 'w') as f:
        for words, labels in test:
            preds = model.predict(words)
            for word, label, pred in zip(words, labels, preds):
                f.write(word + " " + label + " " + pred)
                f.write("\n")

            f.write("\n")


if __name__ == "__main__":
    main()
