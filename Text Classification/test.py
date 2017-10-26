from gtnlplib import preproc
reload(preproc); #terminal semicolon suppresses output

# this will not work until you implement it
y_tr,x_tr = preproc.read_data('reddit-train.csv', #filename
                                       'subreddit', #label field
                                       preprocessor=preproc.tokenize_and_downcase) #your preprocessor

y_dv,x_dv = preproc.read_data('reddit-dev.csv', #filename
                                       'subreddit', #label field
                                       preprocessor=preproc.tokenize_and_downcase) #your preprocessor
y_te,x_te = preproc.read_data('reddit-test.csv', #filename
                                       'subreddit', #label field
                                       preprocessor=preproc.tokenize_and_downcase) #your preprocessor


corpus_counts = preproc.get_corpus_counts(x_tr)

vocab = [word for word,count in corpus_counts.iteritems() if count > 10]
print len(vocab)
print len(x_tr[0])

x_tr = [{key:val for key,val in x_i.iteritems() if key in vocab} for x_i in x_tr]
x_dv = [{key:val for key,val in x_i.iteritems() if key in vocab} for x_i in x_dv]
x_te = [{key:val for key,val in x_i.iteritems() if key in vocab} for x_i in x_te]

from gtnlplib import perceptron
reload(perceptron)

theta_perc,theta_perc_history = perceptron.estimate_perceptron(x_tr,y_tr,20)