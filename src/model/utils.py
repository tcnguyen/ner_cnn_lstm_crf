import numpy as np

def pad_sequences(sequences, pad_tok=0):
    """ pad a sequence to maximum length in sequence with pad_tok """
    sequence_lengths = [len(s) for s in sequences]
    maxlen = max(sequence_lengths)
    sequence_padded = [list(s) + [pad_tok]*(maxlen-len(s)) for s in sequences]
    return sequence_padded, sequence_lengths


def pad_characters(sequences, pad_tok=0):
    """pad sequences of characters id arrays"""

    # first pad the sentence with empty character array to maximum len
    sequence_padded, _ = pad_sequences(sequences, pad_tok=[])
    word_lengths = [[len(chars) for chars in s] for s in sequence_padded]
    maxlen = np.max(word_lengths)

    # pad each chars array to maximum word length
    sequence_padded = [[chars + [pad_tok]*(maxlen-len(chars)) for chars in s] for s in sequence_padded]
    return sequence_padded, word_lengths


assert pad_sequences([[1,2],[3,4,5]]) == \
    ([[1, 2, 0],[3, 4, 5]], \
    [2, 3])

assert pad_characters([[[1,2], [3,4,5,6]], [[3,4,5], [1,2,3], [3,4]]]) == \
    ([[[1, 2, 0, 0], [3, 4, 5, 6], [0, 0, 0, 0]], \
    [[3, 4, 5, 0], [1, 2, 3, 0], [3, 4, 0, 0]]], \
    [[2, 4, 0], [3, 3, 2]])