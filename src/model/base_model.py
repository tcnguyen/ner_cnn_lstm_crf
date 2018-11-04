import os
import tensorflow as tf

from src.config import Config
from src.logger import logger


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self):
        self.logger = logger
        self.sess = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping
        """
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        """Reload weights into session"""
        self.logger.info("Loading the trained model from %s", dir_model)
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        if not os.path.exists(Config.dir_model):
            os.makedirs(Config.dir_model)
        self.saver.save(self.sess, Config.dir_model)

    def close_session(self):
        self.sess.close()

    def add_summary(self):
        """Defines variables for Tensorboard"""
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(Config.dir_output,
                                                 self.sess.graph)

    def train(self, train, dev):
        """Does training with early stopping and lr exponential decay

        Args:
            train, dev: dataset that yields tuple of (sentence, tags)
        """
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping
        self.add_summary()  # tensorboard

        for epoch in range(Config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                                                           Config.nepochs))

            score = self.run_epoch(train, dev, epoch)
            Config.lr *= Config.lr_decay  # decay learning rate

            # early stopping
            if score > best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score. Saved model.")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= Config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "
                                     "improvement".format(nepoch_no_imprv))
                    break
