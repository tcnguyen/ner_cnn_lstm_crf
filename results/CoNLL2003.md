## Results on CoNLL2003 dataset

Code v1.1

### Parameters:

```
dim_char = 50
nepochs = 20
dropout = 0.4 # keep probability
batch_size = 16
lr_method = "adam"
lr = 0.001
lr_decay = 0.9
clip = -1  # if negative, no clipping
    

# model hyperparameters
conv_filter_size = 3
conv_filters = 32
hidden_size_lstm = 300
```

### Results:

Stop after epoch 17 (log.txt). Best dev loss = 0.50191736 (F1 = 94.61) (train loss 0.2)
F1 on test dataset is 90.56. Note that F1 on test dataset is much smaller than on dev (valid) set, which is the same thing as in the paper.

```
processed 49888 tokens with 5648 phrases; found: 5681 phrases; correct: 5130.
accuracy:  91.84%; (non-O)
accuracy:  98.05%; precision:  90.30%; recall:  90.83%; FB1:  90.56
              LOC: precision:  91.88%; recall:  92.99%; FB1:  92.43  1688
             MISC: precision:  80.69%; recall:  80.34%; FB1:  80.51  699
              ORG: precision:  86.67%; recall:  88.44%; FB1:  87.54  1695
              PER: precision:  96.69%; recall:  95.61%; FB1:  96.14  1599

```

### Remarks:

- Dropout parameters is very important to reduce overfitting and obtain low loss on dev dataset.
- The above result used keep probability = 0.4 or dropout 60%. The same run (every parameters stay the same) using keep_prob = 0.3 (dropout 70%) gives:

```
Epoch 17:   dev loss: 0.5363846      train loss: 0.5543   
Epoch 18:   dev loss: 0.5337068      train loss: 0.5484
Epoch 19:   dev loss: 0.5339107      train loss: 0.5242
Epoch 20:   dev loss: 0.5354759      train loss: 0.5235
```

and seems to underfit.

- We tested removing the dropout on inputs of LSTM (ie the word embeddings + chars convnet outputs) using the same parameters, the model is overfitted and we got lower score on dev and test sets.



## Elmo embeddings

```
processed 49888 tokens with 5648 phrases; found: 5679 phrases; correct: 5184.
accuracy:  92.78%; (non-O)
accuracy:  98.31%; precision:  91.28%; recall:  91.78%; FB1:  91.53
              LOC: precision:  92.55%; recall:  92.39%; FB1:  92.47  1665
             MISC: precision:  81.24%; recall:  82.05%; FB1:  81.64  709
              ORG: precision:  88.72%; recall:  89.95%; FB1:  89.33  1684
              PER: precision:  97.04%; recall:  97.28%; FB1:  97.16  1621

```