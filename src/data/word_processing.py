import re

number_regex = re.compile(r'^[-+]?[0-9]+[\.\,]?[0-9]*$')

assert number_regex.match('1-to-2') is None
assert number_regex.match('1,2') is not None

UNKNOWN_TOKEN = "_UNK_"  # the string value is not imporant
PADDING_TOKEN = "_PAD_"  # the string value is not imporant

def transform_number(word):
    #number = float(word)
    return '0'


def transform_word(word):
    """
    process raw word:
    - lowercase
    - transform number
    """
    word = word.lower()

    if number_regex.match(word):
        word = transform_number(word)
       
    return word

