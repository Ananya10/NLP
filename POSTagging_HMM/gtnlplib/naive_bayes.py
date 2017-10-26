import numpy as np #hint: np.log
import sys
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET
import math

def get_corpus_counts(x, y, label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    dict = []
    for i in range(len(y)):
        if y[i] == label:
            dict.append(x[i])

    corpus_counts = defaultdict(float)

    for bow in dict:
        for key, val in bow.iteritems():
            corpus_counts[key] += val

    return corpus_counts


def estimate_pxy(x, y, label, smoothing, vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    x_new = get_corpus_counts(x, y, label)
    total = sum(x_new.values())
    V = len(vocab)
    total += V * smoothing

    corpus_counts_label = defaultdict(float)

    for word in vocab:
        corpus_counts_label[word] = math.log((x_new[word] + smoothing) / (float)(total))

    return corpus_counts_label

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    #hint: use your solution from pset 1
    labels = set(y)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)  # hint

    vocab = set()
    for doc in x:
        for word in doc.keys():
            vocab.add(word)

    for label in labels:
        newdict = estimate_pxy(x, y, label, smoothing, list(vocab))
        for key, val in newdict.iteritems():
            tuple = (label, key)
            counts[tuple] = val

    for label in y:
        doc_counts[label] += 1

    for key, val in doc_counts.iteritems():
        tuple = (key, OFFSET)
        counts[tuple] = math.log(val / len(y))

    return counts

def estimate_nb_tagger(counters,smoothing):
    """build a tagger based on the naive bayes classifier, which correctly accounts for the prior P(Y)

    :param counters: dict of word-tag counters, from most_common.get_tag_word_counts
    :param smoothing: value for lidstone smoothing
    :returns: classifier weights
    :rtype: defaultdict

    """
    # hint: call estimate_nb, then modify the output
    weights = []
    labels = []

    for key in counters.keys():
        weights.append(counters[key])
        labels.append(key)

    estimated_weights = estimate_nb(weights,labels,smoothing)

    doc_counts = defaultdict(float)

    for label in labels:
        doc_counts[label] = sum(counters[label].values())

    total_sum = sum(doc_counts.values())

    for key, val in doc_counts.iteritems():
        tuple = (key, OFFSET)
        estimated_weights[tuple] = math.log(val / float(total_sum))

    return estimated_weights
