from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector,argmax

def perceptron_update(x,y,weights,labels):
    """compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    y_hat, scores = predict(x, weights, labels)
    f_x_y = make_feature_vector(x, y)
    f_x_y_hat = make_feature_vector(x, y_hat)

    update = defaultdict(float)

    diffKeys = set(f_x_y.keys()) - set(f_x_y_hat.keys())

    for key in diffKeys:
        update[key] = f_x_y.get(key)

    diffKeys = set(f_x_y_hat.keys()) - set(f_x_y.keys())

    for key in diffKeys:
        update[key] = 0.0 - f_x_y_hat.get(key)

    return update


def estimate_perceptron(x,y,N_its):
    """estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in xrange(N_its):

        for x_i,y_i in zip(x,y):

            new_weight = perceptron_update(x_i,y_i,weights,labels)

            for key in new_weight.keys():
                weights[key] = weights[key] + new_weight[key]

        weight_history.append(weights.copy())
    return weights, weight_history


def estimate_avg_perceptron(x, y, N_its):
    """estimate averaged perceptron classifier

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    w_sum = defaultdict(float)  # hint
    avg_weights = defaultdict(float)
    weight_history = []

    t = 1.0  # hint
    for it in xrange(N_its):
        for x_i, y_i in zip(x, y):
            weight_new = perceptron_update(x_i, y_i, avg_weights, labels)

            for key in weight_new.keys():
                w_sum[key] = w_sum[key] + weight_new[key] * t
                avg_weights[key] = avg_weights[key] + weight_new[key]

            t = t + 1
        weight_history.append(avg_weights.copy())

    for key in w_sum.keys():
        avg_weights[key] = avg_weights[key] - (w_sum[key] / t)

    return avg_weights, weight_history