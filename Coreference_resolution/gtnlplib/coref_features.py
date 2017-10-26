import itertools
import coref_rules
from nltk import wordnet
import math

# useful?
pronoun_list=['it','he','she','they','this','that']
poss_pronoun_list=['its','his','her','their']
oblique_pronoun_list=['him','her','them']
def_list=['the','this','that','these','those']
indef_list=['a','an','another']

# d3.1
def minimal_features(markables,a,i):
    """Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: features
    :rtype: dict

    """
    f = dict()
    ## your code here

    if a == i:
        f['new-entity'] = 1.0

    else:
        if coref_rules.exact_match(markables[a],markables[i]):
            f['exact-match'] = 1

        if coref_rules.match_last_token(markables[a],markables[i]):
            f['last-token-match'] = 1

        if coref_rules.match_on_content(markables[a],markables[i]):
            f['content-match'] = 1

        # if (markables[a]['start_token'] >= markables[i]['start_token'] and markables[a]['start_token'] <= markables[i]['end_token']):
        #     f['crossover'] = 1
        # if (markables[a]['end_token'] >= markables[i]['start_token'] and markables[a]['end_token'] <= markables[i]['end_token']):
        #     f['crossover'] = 1
        if not markables[a]['start_token'] > markables[i]['end_token'] and not markables[a]['end_token'] < markables[i]['start_token']:
            f['crossover'] = 1

    ## use functions from coref_rules
    return f

# deliverable 3.5
def distance_features(x,a,i,
                      max_mention_distance=10,
                      max_token_distance=10):
    """compute a set of distance features for antecedent a and mention i

    :param x: markable list for document
    :param a: antecedent index
    :param i: mention index
    :param max_mention_distance: upper limit on mention distance
    :param max_token_distance: upper limit on token distance
    :returns: feature dict
    :rtype: dict

    """
    f = dict()
    ## your code here

    if a==i:
        return f

    ant_markable = x[a]
    cur_markable = x[i]

    count = int(math.fabs(a-i))
    if count > max_mention_distance:
        count = max_mention_distance

    mention_str = 'mention-distance-'+str(count)
    f[mention_str] = 1

    st_index = cur_markable['start_token']
    end_index = ant_markable['end_token']

    token_dist = int(math.fabs(end_index - st_index))

    if token_dist > max_token_distance:
        token_dist = max_token_distance

    token_str = 'token-distance-'+ str(token_dist)
    f[token_str] = 1

    return f
    
###### Feature combiners

# deliverable 3.6
def make_feature_union(feat_func_list):
    """return a feature function that is the union of the feature functions in the list

    :param feat_func_list: list of feature functions
    :returns: feature function
    :rtype: function

    """
    def f_out(x,a,i):
        # your code here
        list_dict = []
        result = {}

        for feat_func in feat_func_list:
            list_dict.append(feat_func(x,a,i))

        for element in list_dict:
            result.update(element)

        return result
    return f_out

# deliverable 3.7
def make_feature_cross_product(feat_func1,feat_func2):
    """return a feature function that is the cross-product of the two feature functions

    :param feat_func1: a feature function
    :param feat_func2: a feature function
    :returns: another feature function
    :rtype: function

    """
    def f_out(x,a,i):
        dict1 = feat_func1(x,a,i)
        dict2 = feat_func2(x,a,i)
        dict = {}

        for key1 in dict1.keys():
            for key2 in dict2.keys():
                merge_key = key1+'-'+key2
                dict[merge_key] = 1

        return dict
    return f_out

def minimal_bakeoff_features(markables,a,i):
    """Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: features
    :rtype: dict

    """
    f = dict()
    ## your code here

    if a == i:
        f['new-entity'] = 1.0

    else:
        if coref_rules.exact_match(markables[a],markables[i]):
            f['exact-match'] = 1

        if coref_rules.match_last_token(markables[a],markables[i]):
            f['last-token-match'] = 1

        if coref_rules.match_on_content(markables[a],markables[i]):
            f['content-match'] = 1

        if not markables[a]['start_token'] > markables[i]['end_token'] and not markables[a]['end_token'] < markables[i]['start_token']:
            f['crossover'] = 1

        str1 = ' '.join(markables[a]['string']).strip()
        str2 = ' '.join(markables[i]['string']).strip()

        if str1 in coref_rules.pronouns:
            f['pron-a'] = 1
        if str2 in coref_rules.pronouns:
            f['pron-i'] = 1

    ## use functions from coref_rules
    return f

# def make_feature_bakeoff_cross_product(feat_func1,feat_func2):
#     """return a feature function that is the cross-product of the two feature functions
#
#     :param feat_func1: a feature function
#     :param feat_func2: a feature function
#     :returns: another feature function
#     :rtype: function
#
#     """
#     def f_out(x,a,i):
#         dict1 = feat_func1(x,a,i)
#         dict2 = feat_func2(x,a,i)
#         dict = {}
#
#         for key1 in dict1.keys():
#             for key2 in dict2.keys():
#                 merge_key = key1+'-'+key2
#                 dict[merge_key] = 1
#
#         return dict
#     return f_out

