# Implementation notes

This implementation imitates the implementation in the blog `https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html`.

3 important files:
- `src/config.py`: general, data, model parameters and training parameters configuration.
- `src/model/cnn_lstm_crf_model.py`: implementation of the model.
- `src/data/features_generator`: generate features (i.e np.array) from text sentences and tags

## Model:
- The final word vector is composed of:
    - pretrained word embeddings (glove)
    - output of the convnet on characters
    
There is no hand-made features on word levels or character levels as in the paper since we expect that this can be learnt by the character level cnn.

- Word input vectors (sentences) are feeded to a bi-LSTM model

- The 2 outputs of the bi-LSTM model is concatenated and feeded to a dense layer with n_tags output

- Finally we use a CRF layer to compute the crf log likelihood. (The paper uses sentence level log likelihood which is similar to a normalization factor)

## Preprocessing

- From the train, valid and test data, we generate the list of unique words, tags and characters. 

- Each word will go to a word_processing step:
    - Lowercase
    - Number (integer or float) ⇒ '0'
    
- The vocabulary will contain only words that has pretrained embedding + __PAD__ + __UNK__ tokens. The word embeddings for __PAD__ and __UNK__ tokens will be zeros.

- Other words in the train, valid and test set that is not in the vocabulary will be transformed to __UNK__

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

Unknown words will not have word embeddings, but we should still allow character levels learning. This is a sensitive point which should be implemented correctly since it is easy to transform the unknown words to some unknown token and treat them the same.