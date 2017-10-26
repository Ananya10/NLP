from gtnlplib import constants
from collections import defaultdict

# Deliverable 1.1
def word_feats(words,y,y_prev,m):
    """This function should return at most two features:
    - (y,constants.CURR_WORD_FEAT,words[m])
    - (y,constants.OFFSET)

    Note! You need to handle the case where $m >= len(words)$. In this case, you should only output the offset feature. 

    :param words: list of word tokens
    :param m: index of current word
    :returns: dict of features, containing a single feature and a count of 1
    :rtype: dict

    """
    fv = dict()
    if m < len(words):
        fv[(y, constants.CURR_WORD_FEAT, words[m])] = 1

    fv[(y, constants.OFFSET)] = 1
    return fv

# Deliverable 2.1
def word_suff_feats(words,y,y_prev,m):
    """This function should return all the features returned by word_feats,
    plus an additional feature for each token, indicating the final two characters.

    You may call word_feats in this function.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """
    fv = word_feats(words,y,y_prev,m)
    if m<len(words):
        fv[(y,constants.SUFFIX_FEAT,words[m][-2:])] = 1

    return fv

def word_feats_en(words,y,y_prev,m):
    """This function should return at most two features:
    - (y,constants.CURR_WORD_FEAT,words[m])
    - (y,constants.OFFSET)

    Note! You need to handle the case where $m >= len(words)$. In this case, you should only output the offset feature.

    :param words: list of word tokens
    :param m: index of current word
    :returns: dict of features, containing a single feature and a count of 1
    :rtype: dict

    """
    fv = dict()
    if m < len(words):

        lowercase = words[m].lower()

        if len(words[m]) > 1 and words[m] == len(words[m]) * words[m][0]:

            fv[(y, constants.CURR_WORD_FEAT, words[m][0])] = 1
            fv[(y, constants.CURR_WORD_FEAT, words[m])] = 1

            if lowercase != words[m]:
                fv[(y, constants.CURR_WORD_FEAT, lowercase)] = 1
        else:
            fv[(y, constants.CURR_WORD_FEAT, words[m])] = 1
            if lowercase != words[m]:
                fv[(y, constants.CURR_WORD_FEAT, lowercase)] = 1

    fv[(y, constants.OFFSET)] = 1
    return fv
    
def word_neighbor_feats(words,y,y_prev,m):
    """compute features for the current word being tagged, its predecessor, and its successor.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """
    fv = word_feats(words,y,y_prev,m)

    if m == 0:
        fv[(y,constants.PREV_WORD_FEAT,constants.PRE_START_TOKEN)] = 1
    else:
        fv[(y, constants.PREV_WORD_FEAT, words[m-1])] = 1

    if m < (len(words)-1):
        fv[(y, constants.NEXT_WORD_FEAT, words[m+1])] = 1
    elif m == len(words) - 1 :
        fv[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1

    # hint: use constants.PREV_WORD_FEAT and constants.NEXT_WORD_FEAT
    return fv

def word_neighbor_feats_en(words,y,y_prev,m):
    """compute features for the current word being tagged, its predecessor, and its successor.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """
    fv = word_feats_en(words,y,y_prev,m)

    if m == 0:
        fv[(y,constants.PREV_WORD_FEAT,constants.PRE_START_TOKEN)] = 1
    else:
        lowercase = words[m-1].lower()
        fv[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
        if words[m - 1] != lowercase:
            fv[(y, constants.PREV_WORD_FEAT, lowercase)] = 1

    if m < (len(words)-1):
        lowercase = words[m + 1].lower()
        fv[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
        if words[m + 1] != lowercase:
            fv[(y, constants.NEXT_WORD_FEAT, lowercase)] = 1
    elif m == len(words) - 1 :
        fv[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1

    # hint: use constants.PREV_WORD_FEAT and constants.NEXT_WORD_FEAT
    return fv

    
def word_feats_competitive_en(words,y,y_prev,m):

    #fv = word_feats_en(words, y, y_prev, m)

    #neighbour
    fv = word_neighbor_feats(words,y,y_prev,m)

    #neighbour
    #fv = word_neighbor_feats_en(words,y,y_prev,m)

    if m < len(words):
        word_m = words[m]


    #suffix
    if m<len(words):
        fv[(y,constants.SUFFIX_FEAT,word_m[-2:])] = 1

    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT + "3", word_m[-3:])] = 1

    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT + "4", word_m[-4:])] = 1

    #if m < len(words):
        #fv[(y, constants.SUFFIX_FEAT + "5", word_m[-5:])] = 1

    # prefix
    if m < len(words):
        fv[(y, '--PREFIX--', word_m[0:2])] = 1

    #prev tag
    fv[(y, constants.PREV_TAG_FEAT, y_prev)] = 1


    #bigrams
    if m < len(words)-1 :
        word_m_1 = words[m + 1]
        fv[(y, '--BI-GRAM--', word_m + ' ' + word_m_1)] = 1

        #fv[(y, '--BI-GRAM--', word_m+' '+ word_m_1)] = 1
    elif m == len(words)-1:
        fv[(y, '--BI-GRAM--', word_m + ' ' + constants.POST_END_TOKEN)] = 1


    #check if any character digit
    if m < len(words) and any(char.isdigit() for char in words[m]):
        fv[(y, '--CONTAINS-DIGIT--', word_m)] = 1

    #check if contains hyphen
    if m < len(words) and ('-' in word_m):
        fv[(y, '--CONTAINS-HYPHEN--', word_m)] = 1



    #if m < len(words) and ('.' in word_m) and word_m[len(word_m)-1]!= '.' and not word_m.isupper():
        #fv[(y, '--DOT-NOT-ALL-CAPS--', word_m)] = 1

    # check if contains dot and begins with caps
    #if m < len(words) and '.' in word_m and len(word_m)>1 and not(words[m] == len(words[m]) * words[m][0]):
        #fv[(y, '--CONTAINS-DOT--', word_m)] = 1

    #repeat_char_word_2 = False

    if m < len(words)-2 :
        word_m_2 = words[m + 2]
        word_m_1 = words[m + 1]
        fv[(y, '--TRI-GRAM--', word_m + ' ' +word_m_1 + ' ' + word_m_2)] = 1

            #if m < len(words):
        #if words[m] == len(words[m]) * words[m][0]:
            #fv[(y, '--REMOVE-ALL-REP--', words[m][0])] = 1

    #check contains capital (proper noun)
    #if m < len(words) and (words[m][0].isupper()):
        #fv[(y, '--STARTS-CAPITAL--', words[m])] = 1


    return fv
    
def word_feats_competitive_ja(words,y,y_prev,m):
    # neighbour
    fv = word_neighbor_feats(words, y, y_prev, m)

    # suffix
    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT, words[m][-2:])] = 1

    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT + "3", words[m][-3:])] = 1

    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT + "4", words[m][-4:])] = 1

    # prefix
    if m < len(words):
        fv[(y, '--PREFIX--', words[m][0:2])] = 1

    #if m < len(words):
        #fv[(y, '--PREFIX--3', words[m][0:3])] = 1

    #if m < len(words):
        #fv[(y, '--PREFIX--4', words[m][0:3])] = 1



    # prev tag
    fv[(y, constants.PREV_TAG_FEAT, y_prev)] = 1

    # bigrams
    if m < len(words) - 1:
        fv[(y, '--BI-GRAM--', words[m] + ' ' + words[m + 1])] = 1
    elif m == len(words) - 1:
        fv[(y, '--BI-GRAM--', words[m] + ' ' + constants.POST_END_TOKEN)] = 1

    if m < len(words)-2 :
        fv[(y, '--TRI-GRAM--', words[m] + ' ' + words[m+1] + ' ' + words[m+2])] = 1


    # check if any character digit
    if m < len(words) and any(char.isdigit() for char in words[m]):
        fv[(y, '--CONTAINS-DIGIT--', words[m])] = 1

    # check if contains hyphen
    if m < len(words) and ('-' in words[m]):
        fv[(y, '--CONTAINS-HYPHEN--', words[m])] = 1

    return fv

def hmm_feats(words,y,y_prev,m):
    fv = dict()

    if m < len(words):
        fv[(y,constants.CURR_WORD_FEAT,words[m])] = 1

    fv[(y,constants.PREV_TAG_FEAT,y_prev)] = 1

    return fv

def hmm_feats_competitive_en(words,y,y_prev,m):
    # hmm_feats
    fv = hmm_feats(words, y, y_prev, m)

    # suffix
    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT, words[m][-2:])] = 1

    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT + "3", words[m][-3:])] = 1

    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT + "4", words[m][-4:])] = 1

    # prefix
    if m < len(words):
        fv[(y, '--PREFIX--', words[m][0:2])] = 1

    # bigrams
    if m < len(words) - 1:
        fv[(y, '--BI-GRAM--', words[m] + ' ' + words[m + 1])] = 1
    elif m == len(words) - 1:
        fv[(y, '--BI-GRAM--', words[m] + ' ' + constants.POST_END_TOKEN)] = 1

    # neighbours
    if m == 0:
        fv[(y,constants.PREV_WORD_FEAT,constants.PRE_START_TOKEN)] = 1
    else:
        fv[(y, constants.PREV_WORD_FEAT, words[m-1])] = 1

    if m < (len(words)-1):
        fv[(y, constants.NEXT_WORD_FEAT, words[m+1])] = 1
    elif m == len(words) - 1 :
        fv[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1

    # check if any character digit
    if m < len(words) and any(char.isdigit() for char in words[m]):
        fv[(y, '--CONTAINS-DIGIT--', words[m])] = 1

    # check if contains hyphen
    if m < len(words) and ('-' in words[m]):
        fv[(y, '--CONTAINS-HYPHEN--', words[m])] = 1

    if m < len(words) - 2:
        fv[(y, '--TRI-GRAM--', words[m] + ' ' + words[m + 1] + ' ' + words[m + 2])] = 1

    return fv

# def hmm_feats_competitive_ja(words,y,y_prev,m):
#     # hmm_feats
#     fv = hmm_feats(words, y, y_prev, m)
#
#     # suffix
#     if m < len(words):
#         fv[(y, constants.SUFFIX_FEAT, words[m][-2:])] = 1
#         fv[(y, '--SUFFIX3--', words[m][-3:])] = 1
#         fv[(y, '--SUFFIX3--', words[m][-4:])] = 1
#
#     # prefix
#     if m < len(words):
#         fv[(y, '--PREFIX--', words[m][0:2])] = 1
#
#     # bigrams
#     if m < len(words) - 1:
#         fv[(y, '--BI-GRAM--', words[m] + ' ' + words[m + 1])] = 1
#     elif m == len(words) - 1:
#         fv[(y, '--BI-GRAM--', words[m] + ' ' + constants.POST_END_TOKEN)] = 1
#
#     # neighbours
#     if m == 0:
#         fv[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
#     else:
#         fv[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1
#
#     if m < (len(words) - 1):
#         fv[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
#     elif m == len(words) - 1:
#         fv[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1
#
#     #trigrams
#     if m < len(words)-2 :
#         fv[(y, '--TRI-GRAM--', words[m] + ' ' + words[m+1] + ' ' + words[m+2])] = 1
#
#         # check if any character digit
#     if m < len(words) and any(char.isdigit() for char in words[m]):
#         fv[(y, '--CONTAINS-DIGIT--', words[m])] = 1
#
#         # check if contains hyphen
#     if m < len(words) and ('-' in words[m]):
#         fv[(y, '--CONTAINS-HYPHEN--', words[m])] = 1
#
#     return fv


def hmm_feats_competitive_ja(words,y,y_prev,m):
    # hmm_feats
    fv = hmm_feats(words, y, y_prev, m)

    # suffix
    if m < len(words):
        fv[(y, constants.SUFFIX_FEAT, words[m][-2:])] = 1
        fv[(y, '--SUFFIX3--', words[m][-3:])] = 1
        fv[(y, '--SUFFIX4--', words[m][-4:])] = 1

    # prefix
    if m < len(words):
        fv[(y, '--PREFIX--', words[m][0:2])] = 1
        #fv[(y, '--PREFIX3--', words[m][0:3])] = 1
        #fv[(y, '--PREFIX4--', words[m][0:4])] = 1

    # bigrams
    if m < len(words) - 1:
        fv[(y, '--BI-GRAM--', words[m] + ' ' + words[m + 1])] = 1
    elif m == len(words) - 1:
        fv[(y, '--BI-GRAM--', words[m] + ' ' + constants.POST_END_TOKEN)] = 1

    # neighbours
    if m == 0:
        fv[(y, constants.PREV_WORD_FEAT, constants.PRE_START_TOKEN)] = 1
    else:
        fv[(y, constants.PREV_WORD_FEAT, words[m - 1])] = 1

    if m < (len(words) - 1):
        fv[(y, constants.NEXT_WORD_FEAT, words[m + 1])] = 1
    elif m == len(words) - 1:
        fv[(y, constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN)] = 1

    #trigrams
    if m < len(words)-2 :
        fv[(y, '--TRI-GRAM--', words[m] + ' ' + words[m+1] + ' ' + words[m+2])] = 1

    return fv

def condenseStr(str):

    condensed = str[0]+str[1]

    for c in str[1:]:
        if c == str[0] and c == str[1]:
            str = str[1:]
        else:
            condensed = condensed + c
            str = str[1:]
    return condensed
