import operator
from collections import defaultdict, Counter
from gtnlplib.constants import START_TAG,END_TAG, TRANS, EMIT
import numpy as np

def argmax(scores):
    """Find the key that has the highest value in the scores dict"""
    return max(scores.iteritems(),key=operator.itemgetter(1))[0]

def viterbi_step(tag, m, words, feat_func, weights, prev_scores):
    """
    Calculate the best path score and back pointer for a given node in the trellis

    :param tag: The tag for which we want to calculate the best path
    :param m: index of the token for which we want to calculate the best tag
    :param words: the list of tokens to tag
    :param feat_func: A function of (words, curr_tag, prev_tag, curr_index) that produces features 
    :param weights: A defaultdict that maps features to numeric score. Should not key error for indexing into keys that do not exist.
    :param prev_scores: a dict, in which keys are tags for token m-1 and values are viterbi scores
    :returns: tuple of (best_score, best_feature), where
        best_score   -- The highest score of any sequence of tags
        best_feature -- The feature in the previous layer of the trellis corresponding
            to the best score

    :rtype: tuple

    """
    scores = {}
    for key in prev_scores.keys():
        value = prev_scores[key]
        feats = feat_func(words, tag, key, m)

        transition_prob = weights[(tag, key, TRANS)] * feats[(tag, key, TRANS)]
        emission_prob = weights[(tag, words[m], EMIT)] * feats[(tag, words[m], EMIT)]
        ans = value + transition_prob + emission_prob
        scores[key] = ans

    best_score = max(scores.values())
    best_tag = argmax(scores)

    return best_score, best_tag

def build_trellis(tokens,feat_func,weights,all_tags):
    """Construct a trellis for the hidden Markov model. Output is a list of dicts.

    :param tokens: list of word tokens to be tagged
    :param feat_func: feature function (words, tag, prev_tag, index)
    :param weights: defaultdict of weights
    :param all_tags: list/set of all possible tags
    :returns: list of dicts, length = len(words)
    first dict should represent score from start to token 1, 
    then score from token 1 to token 2,
    etc until token M
    :rtype: list of dicts

    """
    trellis = [None]*(len(tokens))

    # build the first column separately
    trellis[0] = None # your code here
    dictionary = {}
    prev_scores = {}

    for tag in all_tags:

        feats = feat_func(tokens, tag, START_TAG, 0)
        transition_prob = weights[(tag, START_TAG, TRANS)] * feats[(tag, START_TAG, TRANS)]
        emission_prob = weights[(tag, tokens[0], EMIT)] * feats[(tag, tokens[0], EMIT)]
        val = transition_prob + emission_prob
        dictionary[tag] = (val,START_TAG)
        prev_scores[tag] = val
    trellis[0] = dictionary
    
    # iterate over the remaining columns
    for m in range(1,len(tokens)):
        trellis[m] = None # your code here (call viterbi_step)
        dictionary = {}
        new_scores = {}
        for tag in all_tags:
            val = viterbi_step(tag, m, tokens, feat_func, weights, prev_scores)
            dictionary[tag] = val
            new_scores[tag] = val[0]
        prev_scores = new_scores
        trellis[m] = dictionary
        
    return trellis


def viterbi_tagger(tokens,feat_func,weights,all_tags):
    """Tag the given words using the viterbi algorithm
        Parameters:
        words     -- A list of tokens to tag
        feat_func -- A function of (words, curr_tag, prev_tag, curr_index)
        that produces features
        weights   -- A defaultdict that maps features to numeric score. Should
        not key error for indexing into keys that do not exist.
        all_tags  -- A set of all possible tags

        Returns:
        tags       -- The highest scoring sequence of tags (list of tags s.t. tags[i]
        is the tag of words[i])
        best_score -- The highest score of any sequence of tags
    """

    
    trellis = build_trellis(tokens,feat_func,weights,all_tags)

    # Step 1: find last tag and best score
    final_scores = {} #your code here

    last_trellis  = trellis[-1]
    for key in last_trellis:
        val, prev_tag = last_trellis[key]
        feats = feat_func(tokens, END_TAG, key, len(tokens))
        transition_prob = weights[(END_TAG, key, TRANS)] * feats[(END_TAG, key, TRANS)]
        final_scores[key] = val + transition_prob

    last_tag = argmax(final_scores)
    best_score = max(final_scores.values())

    # Step 2: walk backwards through trellis to find best tag sequence
    output = [last_tag] # keep
    for m,v_m in enumerate(reversed(trellis[1:])): #keep

        tag = v_m[last_tag][1]
        output.append(tag)
        last_tag = tag

    output = list(reversed(output))
    return output,best_score

