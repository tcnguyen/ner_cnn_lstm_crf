import numpy as np
import os
import tensorflow as tf

from src.data.utils import pad_sequences
from src.model.base_model import BaseModel
from src.data.features_generator import FeaturesGenerator
from src.config import Config

Progbar = tf.keras.utils.Progbar


class CharCNNLSTMCRFModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self):
        super(CharCNNLSTMCRFModel, self).__init__()
        self.features_generator = FeaturesGenerator()

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, features, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            features: list of ([char ids], word_id) for each word in the sentence
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        char_ids, word_ids = zip(*features)
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                               nlevels=2)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        feed[self.char_ids] = char_ids
        feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        Final word embeddings is the concatenation of:
        - Pretrained word embeddings like glove, word2vec
        - Output of characters convnet
        """
        with tf.variable_scope("words"):
            with np.load(Config.DATA_PATHS['word_embeddings']) as data:
                embeddings_table = data["embeddings"]

            _word_embeddings = tf.Variable(
                embeddings_table,
                name="_word_embeddings",
                dtype=tf.float32,
                trainable=Config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            # get char embeddings matrix
            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[self.features_generator.nchars, Config.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                     self.char_ids, name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            self.char_embeddings = tf.reshape(char_embeddings,
                                              shape=[s[0]*s[1], s[-2], Config.dim_char])

            self.char_conv = tf.layers.conv1d(
                self.char_embeddings,
                filters=Config.conv_filters,
                kernel_size=(Config.conv_filter_size),
                strides=1,
                padding='SAME',
                data_format='channels_last',
                name="char_conv"
            )

            self.char_max = tf.reduce_max(self.char_conv, axis=-2)

            self.char_output = tf.reshape(self.char_max,
                                          shape=[s[0], s[1], Config.conv_filters])
            #word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

            word_embeddings = tf.concat(
                [word_embeddings, self.char_output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)
        #self.word_embeddings = word_embeddings

    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(Config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(Config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2*Config.hidden_size_lstm, self.features_generator.ntags])

            b = tf.get_variable("b", shape=[self.features_generator.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*Config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(
                pred, [-1, nsteps, self.features_generator.ntags])

    def add_loss_op(self):
        """Defines the loss"""

        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(Config.lr_method, self.lr, self.loss,
                          Config.clip)
        self.initialize_session()  # now self.sess is defined and vars are init

    def predict_batch(self, features):
        """
        Args:
            features: list of char_ids and word_ids

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(features, dropout=1.0)

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params = self.sess.run(
            [self.logits, self.trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            loss on development set
        """
        # progbar stuff for logging
        batch_size = Config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (features, labels) in enumerate(self.features_generator.minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(features, labels, Config.lr,
                                       Config.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        # loss over dev
        dev_losses = []
        for i, (features, labels) in enumerate(self.features_generator.minibatches(dev, batch_size)):
            fd, _ = self.get_feed_dict(features, labels, dropout=1.0)

            dev_loss = self.sess.run(self.loss, feed_dict=fd)
            dev_losses.append(dev_loss)

        self.logger.info("dev loss: %s", np.mean(dev_losses))

        # metrics = self.run_evaluate(dev)
        # msg = " - ".join(["{} {:04.2f}".format(k, v)
        #                   for k, v in metrics.items()])
        # self.logger.info(msg)

        # return metrics["f1"]
        return np.mean(dev_losses)

    def predict(self, words):
        """Returns list of tags

        Args:
            words: list of words (string)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        features = self.features_generator.compute_features(words)
        pred_ids, _ = self.predict_batch([features])
        preds = [self.features_generator.idx_to_tag[idx]
                 for idx in list(pred_ids[0])]

        return preds
