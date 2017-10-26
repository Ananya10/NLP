import nltk
import pandas as pd
from collections import Counter
from nltk import word_tokenize, sent_tokenize

def tokenize_and_downcase(string,vocab=None):
    """for a given string, corresponding to a document:
    - tokenize first by sentences and then by word
    - downcase each token
    - return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    lines = sent_tokenize(string)
    tokens = []
    for line in lines:
        if line.strip():
            tokens = tokens + word_tokenize(line)

    lower_tokens = [token.lower() for token in tokens]

    bow = Counter()

    for token in lower_tokens:
        bow[token] += 1

    return bow


### Helper code

def read_data(csvfile,labelname,preprocessor=lambda x : x):
    # note that use of utf-8 encoding to read the file
    df = pd.read_csv(csvfile,encoding='utf-8')
    return df[labelname].values,[preprocessor(string) for string in df['text'].values]

def get_corpus_counts(list_of_bags_of_words):
    counts = Counter()
    for bow in list_of_bags_of_words:
        for key,val in bow.iteritems():
            counts[key] += val
    return counts

### Secret bakeoff code
def custom_preproc(string):
    """for a given string, corresponding to a document, tokenize first by sentences and then by word; downcase each token; return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    stop_words = ["a","an","and","are","as","at","be","by","for","from",
                  "has","he","in","is","it","its","of","on","that","the",
                  "to","was","were","will","with"];
    stop_words_set = set(stop_words)

    lines = sent_tokenize(string)
    tokens = []
    for line in lines:
        if line.strip():
            tokens = tokens + word_tokenize(line)

    lower_tokens = [token.lower() for token in tokens]

    bow = Counter()

    for token in lower_tokens:
        if token.isalpha() and token not in stop_words_set:
            bow[token] += 1

    return bow

