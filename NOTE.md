# Implementation notes

This implementation is inspired from the one described in the blog: `https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html`.

**3 important files**:
- `src/config.py`: all configurable parameters are put here.
- `src/model/cnn_lstm_crf_model.py`: implementation of the model.
- `src/data/features_generator`: features generation from text sentences and tags.

## Model:
- The final word vector is composed of:
    - pretrained word embeddings (glove)
    - output of the convnet on characters
    
There is no hand-made features on word since we expect that they can be learnt by the character convnet.

- Those word input vectors are feeded to a bi-LSTM model.

- The 2 outputs of the bi-LSTM model is concatenated and feeded to a dense layer with `ntags` outputs.

- Finally we use a è `CRF` layer to compute the crf log likelihood (which is the negative of the loss).

### Dropout:

- We applied dropout on the outputs of the LTSM layers and after the word vectors (i.e glove embeddings + chars convnet = the input of the LSTM)
- The paper mentioned that applying dropout on the input of the LSTM *seems* to have adverse effect (section 2.6.4) so we want to try both options.

## Preprocessing

- From the train, valid and test data, we generate the list of unique words, tags and characters. 

- Each word will go to a processing step:
    - Lowercase
    - Number (integer or float) ⇒ '0'
    
- The vocabulary will contain only those processed words which **have pretrained embedding** + __PAD__ + __UNK__ tokens. The word embeddings for __PAD__ and __UNK__ tokens will be zeros. During train time and test time, words that are not in the vocabulary (i.e no pretrained embedding available) will be transformed to __UNK__ (but we still keep the word characters to feed through the convnet since they are very useful even if the words are unknown).

## Features format:

X = char_ids, word_ids  
y = tag_ids


Example of a batch of 2 sentences:

```
[["I", "love", "Paris"], 
  ["Peter", "is", "not", "french"]]
```

The input data which will be feeded to our model will be:

`word_ids`: (2,4) with 4 = max sentence length, 0 padding at the end

```
[ [14000,   18367,   19707,   0        ],
  [7580,     19917,   2955,     17178] ],
```

`char_ids`:  (2,4,6)  with 4 = max sentence length and 6 = max word length ("french"), 0 padding at the end for all the characters 

```
[[[27, 0, 0, 0, 0, 0],
   [5, 17, 56, 79, 0, 0],
   [35, 2, 36, 52, 15, 0],
   [0, 0, 0, 0, 0, 0]],
  [[35, 79, 82, 79, 36, 0],
   [52, 15, 0, 0, 0, 0],
   [71, 17, 82, 0, 0, 0],
   [62, 36, 79, 71, 13, 19]]],
```

## Note on unknown words:

Unknown words will not have word embeddings, but we should still allow character levels learning. This is an important point which should be implemented correctly.