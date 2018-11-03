## Results on CoNLL2003 dataset

Code v1.1

### Parameters:

```
dim_char = 50
nepochs = 20
dropout = 0.5
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

Early stopping happened with best dev loss = 0.55268866 (F1 = 94.09)


F1 on test dataset is 90.08. Note that F1 on test dataset is much smaller than on dev (valid) set, which is the same thing as in the paper.

```
processed 49888 tokens with 5648 phrases; found: 5689 phrases; correct: 5106.
accuracy:  91.43%; (non-O)
accuracy:  98.01%; precision:  89.75%; recall:  90.40%; FB1:  90.08
              LOC: precision:  91.53%; recall:  93.29%; FB1:  92.40  1700
             MISC: precision:  78.95%; recall:  79.63%; FB1:  79.29  708
              ORG: precision:  87.01%; recall:  87.54%; FB1:  87.27  1671
              PER: precision:  95.47%; recall:  95.05%; FB1:  95.26  1610
```