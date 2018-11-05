# https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub

# https://github.com/allenai/bilm-tf

import tensorflow_hub as hub

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

tokens_input = [["the", "cat", "is", "on", "the", "mat"],
["dogs", "are", "in", "the", "fog", ""]]

tokens_length = [6, 5]

embeddings = elmo(
    inputs={
    "tokens": tokens_input,
    "sequence_len": tokens_length
    }, signature="tokens", as_dict=True)["elmo"]


import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(embeddings)
