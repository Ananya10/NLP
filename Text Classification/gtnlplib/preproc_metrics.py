def get_token_type_ratio(vocabulary):
    """compute the ratio of tokens to types

    :param vocabulary: a Counter of words and their frequencies
    :returns: ratio of tokens to types
    :rtype: float

    """
    return sum(vocabulary.values())/float(len(vocabulary.values()));

def type_frequency(vocabulary, k):
    """compute the number of words that occur exactly k times

    :param vocabulary: a Counter of words and their frequencies
    :param k: desired frequency
    :returns: number of words appearing k times
    :rtype: int

    """
    return vocabulary.values().count(k)

def unseen_types(first_vocab, second_vocab):
    """compute the number of words that appear in the second vocab but not in the first vocab

    :param first_vocab: a Counter of words and their frequencies in one dataset
    :param second_vocab: a Counter of words and their frequencies in another dataset
    :returns: number of words that appear in the second dataset but not  in the first dataset
    :rtype: int

    """
    first_vocab_set = set(first_vocab.keys())
    second_vocab_set = set(second_vocab.keys())
    return len(second_vocab_set - first_vocab_set)


