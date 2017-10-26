import numpy as np
from collections import defaultdict
import coref


# deliverable 3.2
def mention_rank(markables, i, feats, weights):
    """ return top scoring antecedent for markable i

    :param markables: list of markables
    :param i: index of current markable to resolve
    :param feats: feature function
    :param weights: weight defaultdict
    :returns: index of best scoring candidate (can be i)
    :rtype: int

    """
    ## hide
    top_score = float('-inf')
    final_index = -1

    for index in range(i + 1):
        features = feats(markables, index, i)
        value = 0.0
        for feature in features.keys():
            value += features[feature] * weights[feature]

        if value > top_score:
            final_index = index
            top_score = value

    return final_index


# deliverable 3.3
def compute_instance_update(markables, i, true_antecedent, feats, weights):
    """Compute a perceptron update for markable i.
    This function should call mention_rank to determine the predicted antecedent,
    and should make an update if the true antecedent and predicted antecedent refer to different entities

    Note that if the true and predicted antecedents refer to the same entity, you should not
    make an update, even if they are different.

    :param markables: list of markables
    :param i: current markable
    :param true_antecedent: ground truth antecedent
    :param feats: feature function
    :param weights: defaultdict of weights
    :returns: dict of updates
    :rtype: dict

    """
    # keep
    pred_antecedent = mention_rank(markables, i, feats, weights)
    entities = coref.get_entities(markables)

    ## possibly useful
    # print i,true_antecedent,pred_antecedent
    # print markables[i]#['string']
    # print markables[true_antecedent], feats(markables,true_antecedent,i)
    # print markables[pred_antecedent], feats(markables,pred_antecedent,i)
    # print ""

    pred_val = markables[pred_antecedent]
    true_val = markables[true_antecedent]
    # print 'pred_val:'
    # print pred_val
    # print 'true val:'
    # print true_val

    true_features = {}

    if pred_val['entity'] != true_val['entity']:
        pred_features = defaultdict(float, feats(markables, pred_antecedent, i))
        true_features = defaultdict(float, feats(markables, true_antecedent, i))
        for feature in pred_features.keys():
            true_features[feature] -=  pred_features[feature]

    # else:
    #     for entity in entities:
    #         if i in entity and entity.index(i) >= 1:
    #             if pred_antecedent == i or pred_antecedent not in entity:
    #                 pred_features = defaultdict(float, feats(markables, pred_antecedent, i))
    #                 true_features = defaultdict(float, feats(markables, true_antecedent, i))
    #                 for feature in pred_features.keys():
    #                     true_features[feature] -=  pred_features[feature]
    #             break
    else:
        if pred_antecedent == i and true_antecedent!=pred_antecedent:
            pred_features = defaultdict(float, feats(markables, pred_antecedent, i))
            true_features = defaultdict(float, feats(markables, true_antecedent, i))
            for feature in pred_features.keys():
                true_features[feature] -= pred_features[feature]

    return true_features


# deliverable 3.4
def train_avg_perceptron(markables, features, N_its=20):
    # the data and features are small enough that you can
    # probably get away with naive feature averaging

    weights = defaultdict(float)
    tot_weights = defaultdict(float)
    weight_hist = []
    T = 0.0

    for it in xrange(N_its):
        num_wrong = 0  # helpful but not required to keep and print a running total of errors
        for document in markables:
            ant_true = coref.get_true_antecedents(document)
            for i in range(len(document)):
                update_val = compute_instance_update(document, i, ant_true[i], features, weights)
                # print 'weights:'
                # print update_val
                # print '-----------'
                if len(update_val.keys()) != 0:
                    num_wrong += 1
                    # print i

                for key in update_val.keys():
                    if update_val[key] != 0.0:
                        weights[key] += update_val[key]

                for key in weights.keys():
                    if weights[key] != 0.0:
                        tot_weights[key] += weights[key]

                # print tot_weights
                # print '----------'
                T += 1.0

                # print document
                # pass

        print num_wrong,

        # update the weight history
        weight_hist.append(defaultdict(float))
        for feature in tot_weights.keys():
            weight_hist[it][feature] = tot_weights[feature] / T

    return weight_hist


# helpers
def make_resolver(features, weights):
    return lambda markables: [mention_rank(markables, i, features, weights) for i in range(len(markables))]


def eval_weight_hist(markables, weight_history, features):
    scores = []
    for weights in weight_history:
        score = coref.eval_on_dataset(make_resolver(features, weights), markables)
        scores.append(score)
    return scores