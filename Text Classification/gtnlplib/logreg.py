from scipy.misc import logsumexp #hint
import numpy as np
from collections import defaultdict

from gtnlplib.clf_base import predict, make_feature_vector
from gtnlplib.constants import OFFSET

def compute_py(x,weights,labels):
    """compute probability P(y | x)

    :param x: base features
    :param weights: current weights 
    :param labels: list of all possible labels
    :returns: probability distribution p(y | x), represented as dict {label:p(label|x)}
    :rtype: dict

    """
    y_test, scores = predict(x,weights,labels)
    denom_sum = logsumexp(scores.values())

    final_dict = {}
    for label in labels:
        final_dict[label] = np.exp(scores[label] - denom_sum)
    # hint: you should use clf_base.predict and logsumexp
    return final_dict
    
def estimate_logreg(x,y,N_its,learning_rate=1e-4,regularizer=1e-2,lazy_reg=True):
    """estimate a logistic regression classifier

    :param x: training instances
    :param y: training labels
    :param N_its: number of training iterations
    :param learning_rate: how far to move on the gradient for each instance
    :param regularizer: how much L2 regularization to apply at each update
    :param lazy_reg: whether to do lazy regularization or not
    :returns: dict of feature weights, list of feature weights at each training epoch
    :rtype: dist, list

    """
    weights = defaultdict(float)
    weight_hist = [] #keep a history of the weights after each iteration
    all_labels = set(y)
    
    # this block is for lazy regularization
    ratereg = learning_rate * regularizer
    def regularize(base_feats):
        for base_feat in base_feats:
            for label in all_labels:
                #print "regularizing",(label,base_feat),t,last_update[base_feat],(1. - ratereg) ** (t-last_update[base_feat])
                weights[(label,base_feat)] *= (1. - ratereg) ** (t-last_update[base_feat])
            last_update[base_feat] = t

    t = 0
    last_update = defaultdict(int)

    eeta = learning_rate

    for it in xrange(N_its):

        for i,(x_i,y_i) in enumerate(zip(x,y)): #keep
            t += 1

            # regularization
            if lazy_reg: # lazy regularization is essential for speed
                regularize(x_i) # only regularize features in this instance
            if not lazy_reg: # for testing/explanatory purposes only
                for feat,weight in weights.iteritems():
                    if feat[1] is not OFFSET: # usually don't regularize offset
                        weights[feat] -= ratereg * weight

            p_y = compute_py(x_i,weights,all_labels) #hint

            term2 = make_feature_vector(x_i, y_i)

            for key in term2.keys():
                weights[key] = weights[key] + (term2[key]*eeta)

            for label in all_labels:
                temp = make_feature_vector(x_i, label)
                for key in temp.keys():
                    weights[key] = weights[key] - (temp[key]*eeta*p_y[label])


        print it,
        weight_hist.append(weights.copy()) 

    # if lazy, let regularizer catch up
    if lazy_reg:
        # iterate over base features
        regularize(list(set([f[1] for f in weights.keys() if f[1] is not OFFSET])))

    return weights,weight_hist
