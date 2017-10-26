from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT, OFFSET
from gtnlplib import naive_bayes, most_common
import numpy as np
from collections import defaultdict

def hmm_features(tokens,curr_tag,prev_tag,m):
    """Feature function for HMM that returns emit and transition features

    :param tokens: list of tokens 
    :param curr_tag: current tag
    :param prev_tag: previous tag
    :param i: index of token to be tagged
    :returns: dict of features and counts
    :rtype: dict

    """
    result = defaultdict(int)
    if curr_tag != END_TAG:
        result[(curr_tag, tokens[m], EMIT)] = 1

    result[(curr_tag, prev_tag, TRANS)] = 1

    return result

def compute_HMM_weights(trainfile,smoothing):
    """Compute all weights for the HMM

    :param trainfile: training file
    :param smoothing: float for smoothing of both probability distributions
    :returns: defaultdict of weights, list of all possible tags (types)
    :rtype: defaultdict, list

    """
    # hint: these are your first two lines
    tag_trans_counts = most_common.get_tag_trans_counts(trainfile)
    all_tags = tag_trans_counts.keys()

    # hint: call compute_transition_weights
    trans_weights = compute_transition_weights(tag_trans_counts, smoothing)

    # hint: set weights for illegal transitions to -np.inf
    for key in all_tags:
        trans_weights[(key,END_TAG,TRANS)] = -np.inf
        trans_weights[(START_TAG,key,TRANS)] = -np.inf

    trans_weights[(END_TAG,END_TAG,TRANS)] = -np.inf

    # hint: call get_tag_word_counts and estimate_nb_tagger
    tag_word_counts = most_common.get_tag_word_counts(trainfile)
    word_weights = naive_bayes.estimate_nb_tagger(tag_word_counts, smoothing)
    new_weights = {}

    for key in word_weights:
        new_key = (key[0],key[1],EMIT)
        if key[1] != OFFSET:
            new_weights[new_key] = word_weights[key]

    # hint: Counter.update() combines two Counters
    trans_weights.update(new_weights)

    # hint: return weights, all_tags
    return trans_weights, all_tags


def compute_transition_weights(trans_counts, smoothing):
    """Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag,TRANS)] and weights

    """
    all_tags = trans_counts.keys()
    all_tags.append(END_TAG)

    weights = defaultdict(float)
    for key in trans_counts.keys():
        counter = trans_counts[key]
        total = sum(counter.values())

        for key2 in all_tags:
            if key2 != START_TAG:
                weights[(key2, key, TRANS)] = np.log((counter[key2]+smoothing)/(total+((len(all_tags)-1)*smoothing)))
            else:
                weights[(key2, key, TRANS)] = -np.inf

    return weights






