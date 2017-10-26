from gtnlplib import tagger_base, constants
from collections import defaultdict

def sp_update(tokens,tags,weights,feat_func,tagger,all_tags):
    """compute the structure perceptron update for a single instance

    :param tokens: tokens to tag
    :param tags: gold tags
    :param weights: weights
    :param feat_func: local feature function from (tokens,y_m,y_{m-1},m) --> dict of features and counts
    :param tagger: function from (tokens,feat_func,weights,all_tags) --> tag sequence
    :param all_tags: list of all candidate tags
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    y_hat, score = tagger(tokens, feat_func, weights, all_tags)
    f_x_y = tagger_base.compute_features(tokens, tags,feat_func)
    f_x_y_hat = tagger_base.compute_features(tokens, y_hat,feat_func)

    update = defaultdict(float)
    f_x_y_keys = f_x_y.keys()
    f_x_y_hat_keys = f_x_y_hat.keys()

    total_keys = list(set(f_x_y_keys).union(f_x_y_hat_keys))

    for key in total_keys:
        if key not in f_x_y_hat_keys:
            update[key] += f_x_y.get(key)
        elif key not in f_x_y_keys:
            update[key] -= f_x_y_hat.get(key)
        else:
            update[key]+=(f_x_y.get(key) - f_x_y_hat.get(key))

    return update
    
def estimate_perceptron(labeled_instances,feat_func,tagger,N_its,all_tags=None):
    """Estimate a structured perceptron

    :param labeled instances: list of (token-list, tag-list) tuples, each representing a tagged sentence
    :param feat_func: function from list of words and index to dict of features
    :param tagger: function from list of words, features, weights, and candidate tags to list of tags
    :param N_its: number of training iterations
    :param all_tags: optional list of candidate tags. If not provided, it is computed from the dataset.
    :returns: weight dictionary
    :returns: list of weight dictionaries at each iteration
    :rtype: defaultdict, list

    """
    """
    You can almost copy-paste your perceptron.estimate_avg_perceptron function here. 
    The key differences are:
    (1) the input is now a list of (token-list, tag-list) tuples
    (2) call sp_update to compute the update after each instance.
    """

    # compute all_tags if it's not provided
    if all_tags is None:
        all_tags = set()
        for tokens,tags in labeled_instances:
            all_tags.update(tags)

    # this initialization should make sure there isn't a tie for the first prediction
    # this makes it easier to test your code
    weights = defaultdict(float,
                          {('NOUN',constants.OFFSET):1e-3})

    weight_history = []
    avg_weights = defaultdict(float,weights)

    # the rest is up to you!

    w_sum = defaultdict(float)  # hint

    t = 0.0  # hint
    for it in xrange(N_its):
        for (tokens,tags) in labeled_instances:

            weight_new = sp_update(tokens, tags, avg_weights, feat_func,tagger,all_tags)

            for key in weight_new.keys():
                w_sum[key] = w_sum[key] + weight_new[key] * t
                avg_weights[key] = avg_weights[key] + weight_new[key]

            t = t + 1

        #weight_history.append(avg_weights.copy())
        copy_weight = avg_weights.copy()
        for key in w_sum.keys():
            copy_weight[key] = copy_weight[key] - (w_sum[key] / t)

        weight_history.append(copy_weight)

    for key in w_sum.keys():
        avg_weights[key] = avg_weights[key] - (w_sum[key] / t)

    return avg_weights, weight_history



