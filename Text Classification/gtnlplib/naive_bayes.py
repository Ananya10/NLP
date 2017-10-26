from gtnlplib.preproc import get_corpus_counts
from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict
import math

def get_corpus_counts(x,y,label):
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
        for key,val in bow.iteritems():
            corpus_counts[key] += val

    return corpus_counts
    
def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    x_new  = get_corpus_counts(x,y,label)
    total = sum(x_new.values())
    V = len(vocab)
    total += V*smoothing

    corpus_counts_label = defaultdict(float)


    for word in vocab:
        corpus_counts_label[word] = math.log((x_new[word] + smoothing)/(float)(total))

    return corpus_counts_label
    
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    labels = set(y)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)  # hint

    vocab = set()
    for doc in x:
        for word in doc.keys():
            vocab.add(word)

    for label in labels:
        newdict = estimate_pxy(x,y,label,smoothing,list(vocab))
        for key, val in newdict.iteritems():
            tuple = (label, key)
            counts[tuple] = val

    for label in y:
        doc_counts[label]+=1

    for key, val in doc_counts.iteritems():
        tuple = (key, OFFSET)
        counts[tuple] = math.log(val/len(y))

    return counts
    
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values to try
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    """
    smoother_acc = {}
    labels = set(y_dv)
    for smoother in smoothers:
        theta = estimate_nb(x_tr, y_tr, smoother)
        y_hat = clf_base.predict_all(x_dv, theta, labels)
        smoother_acc[smoother] = evaluation.acc(y_hat, y_dv)

    argmax = lambda x: max(x.iteritems(), key=lambda y: y[1])[0]
    return argmax(smoother_acc), smoother_acc
