### Rule-based coreference resolution  ###########
# Lightly inspired by Stanford's "Multi-pass sieve"
# http://www.surdeanu.info/mihai/papers/emnlp10.pdf
# http://nlp.stanford.edu/pubs/conllst2011-coref.pdf

import nltk

# this may help
pronouns = ['I','me','mine','you','your','yours','she','her','hers','he','him','his','it','its','they','them','their','theirs','this','those','these','that','we','our','us','ours']
downcase_list = lambda toks : [tok.lower() for tok in toks]

############## Pairwise matchers #######################

def exact_match(m_a,m_i):
    """return True if the strings are identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical
    :rtype: boolean

    """
    return downcase_list(m_a['string'])==downcase_list(m_i['string'])

# deliverable 2.2
def exact_match_no_pronouns(m_a,m_i):
    """return True if strings are identical and are not pronouns

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical and are not pronouns
    :rtype: boolean

    """
    s1 = m_a['string']
    s2 = m_i['string']

    str1 = ' '.join(s1).strip()
    str2 = ' '.join(s1).strip()

    if str1 in pronouns:
        return False

    if str2 in pronouns:
        return False

    return downcase_list(s1) == downcase_list(s2)

# deliverable 2.3
def match_last_token(m_a,m_i):
    """return True if final token of each markable is identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :rtype: boolean

    """
    return downcase_list(m_a['string'])[-1] == downcase_list(m_i['string'])[-1]

# deliverable 2.4
def match_last_token_no_overlap(m_a,m_i):
    """

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if final tokens match and strings do not overlap
    :rtype: boolean

    """
    return downcase_list(m_a['string'])[-1] == downcase_list(m_i['string'])[-1] and (m_a['start_token'] > m_i['end_token'] or m_a['end_token'] < m_i['start_token'])
    
# deliverable 2.5
def match_on_content(m_a, m_i):
    """

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if all match on all "content words" (defined by POS tag) and markables do not overlap
    :rtype: boolean

    """
    str1 = downcase_list(m_a['string'])
    str2 = downcase_list(m_i['string'])
    tag1 = m_a['tags']
    tag2 = m_i['tags']
    tag_list = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$', 'CD']

    str1_new = []
    str2_new = []
    for i in range(len(str1)):
        if tag1[i] in tag_list:
            str1_new.append(str1[i])

    for i in range(len(str2)):
        if tag2[i] in tag_list:
            str2_new.append(str2[i])

    # str1_new = ' '.join(str1_new).strip()
    # str2_new = ' '.join(str2_new).strip()

    return str1_new == str2_new and (m_a['start_token'] > m_i['end_token'] or m_a['end_token'] < m_i['start_token'])


########## helper code

def most_recent_match(markables,matcher):
    """given a list of markables and a pairwise matcher, return an antecedent list
    assumes markables are sorted

    :param markables: list of markables
    :param matcher: function that takes two markables, returns boolean if they are compatible
    :returns: list of antecedent indices
    :rtype: list

    """
    antecedents = range(len(markables))
    for i,m_i in enumerate(markables):
        for a,m_a in enumerate(markables[:i]):
            if matcher(m_a,m_i):
                antecedents[i] = a
    return antecedents

def make_resolver(pairwise_matcher):
    """convert a pairwise markable matching function into a coreference resolution system, which generates antecedent lists

    :param pairwise_matcher: function from markable pairs to boolean
    :returns: function from markable list and word list to antecedent list
    :rtype: function

    The returned lambda expression takes a list of words and a list of markables. 
    The words are ignored here. However, this function signature is needed because
    in other cases, we want to do some NLP on the words.

    """
    return lambda markables : most_recent_match(markables,pairwise_matcher)
