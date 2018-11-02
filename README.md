# Setup

```
virtualenv -p python3.6 env
. env/bin/activate
pip install -r requirements.txt
```

## Data

Put `train.txt`, `valid.txt` (testa) and `test.txt` (testb) files in `data\CoNLL2003` folder.

## Config

In `src/config.py` put the path to the glove word embedding file, for example:

```python
EMBEDDING_PATH = '../../word_embeddings/glove.6B.300d.txt'
```

# Steps:

```bash
# preprocess data
python -m src.scripts.preprocessing

# train
python -m src.scripts.train

# generate test prediction
python -m src.scripts.prediction

# compute metrics on test predictions (using python version of conlleval script)
python -m src.scripts.conlleval < test.preds.txt
```