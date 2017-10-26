import coref_features,coref_learning,coref,os
from nltk.tag import pos_tag

dv_dir = os.path.join('data','dev')
tr_dir = os.path.join('data','tr')
te_dir = os.path.join('data','te-hidden-labels')

all_markables,_ = coref.read_dataset(tr_dir,tagger=pos_tag)
all_markables_dev,_ = coref.read_dataset(dv_dir,tagger=pos_tag)
all_markables_te,_ = coref.read_dataset(te_dir,tagger=pos_tag)

theta_simple = coref_learning.train_avg_perceptron([all_markables[3][:10]],coref_features.minimal_features,N_its=2)
print theta_simple