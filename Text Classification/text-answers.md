# Deliverable 1.3

Why do you think the type-token ratio is lower for the dev data as compared to the training data?

(Yes the dev set is smaller; why does this impact the type-token ratio?)

Answer 1.3: The training data set is much larger than the dev dataset. But with each addition of a new document to a dataset, the number of words added to the vocabulary is comparatively smaller than the total number of words in the document since there are alot of words like "a", "the", "is", "it", "of" etc which are common to all documents. Hence the token to tpe ratio which is the total number of words in the vocabulary / unique words in the vocabulary will be higher for training data as compared to dev data.


# Deliverable 3.5

Explain what you see in the scatter plot of weights across different smoothing values.

Answer 3.5: 

# Deliverable 6.2

Now compare the top 5 features for logistic regression under the largest regularizer and the smallest regularizer.
Paste the output into ```text_answers.md```, and explain the difference. (.4/.2 points)

Answer 6.2: The top 5 features for the largest value of the regularizer returns words which are common accross all documents of the label 'like' 'of', 'is' etc. The smallest regularizer value returns words more specific to the label as the top 5 features. For example: for regularizer = 1e-1, top 5 features for science are: **OFFSET**', 'corn', 'is', 'of', 'research' whereas for regularizer = 1e-4, top 5 features for science are: 'psychopath', 'research', 'pollution', 'ebv', 'evolution'. Hence the higher regularizer value gives low variance but higher bias and the lower regularizer gives lower bias but probably a higher variance


# Deliverable 7.2

Explain the new preprocessing that you designed: why you thought it would help, and whether it did.

Answer 7.2: In the new preprocessing function, I have removed non-alphabetic characters as well as a list of common English stopwords ("a", "the", "is", "it", "of" etc) from the bag of words because punctuation and these stopwords appear in documents belonging to all of the labels and hence the prediction of a label given a document should not be dependent on these words. After the new pre-processing function, the accuracy of prediction on the dev set increased to 78.2%(0.78200000000000003)

# Deliverable 8

Describe the research paper that you have chosen.

- What are the labels, and how were they obtained?
- Why is it interesting/useful to predict these labels?  
- What classifier(s) do they use, and the reasons behind their choice? Do they use linear classifiers like the ones in this problem set?
- What features do they use? Explain any features outside the bag-of-words model, and why they used them.
- What is the conclusion of the paper? Do they compare between classifiers, between feature sets, or on some other dimension? 
- Give a one-sentence summary of the message that they are trying to leave for the reader.

Describe the research paper that you have chosen
Answer:
Document Type Classification in Online Digital Libraries - AAAI 2016 - Cornelia Caragea, JianWu, Sujatha Das Gollapalli, and C. Lee Giles
The research paper is about document type classification in online digital libraries. The accuracy of the classifiers depend on the choice of feature representation. The paper proposes some novel features that result in higher acuuracy of document type classification than the common classifiers trained using bag of words as well as rule based classifiers. 
 
What are the labels, and how were they obtained?
Answer: The labels are: Books, Slides, Theses, Papers, CVs, Others. These labels were chosen because online digital libraries such as Google Scholar, CiteSeerx, ACL Anthology, ArnetMiner, and PubMed that store scientific documents or their metadata classify documents into these categories

What classifier(s) do they use, and the reasons behind their choice? Do they use linear classifiers like the ones in this problem set?
Answer: The authors experiemented with different feature sets on various classifiers like Random Forest(RF), Decision Trees (DT), Na¨ıve Bayes Multinomial(NBM), Na¨ıve Bayes (NB), and Support Vector Machines with a linear kernel (SVM). They have used classifiers which have implementation in commonly used libraries like Weka so that they can evaluate the impact of having different feature sets on the accuracy of the classifier easily.

What features do they use? Explain any features outside the bag-of-words model, and why they used them.
Answer: Apart from the bag of words model, some other features used by the authors of the paper are:
1. File specific features - characteristics like file size, number of pages in the document. Research papers often contain less pages than books or theses and can be used to differentiate between the type of document.
2. Text or document specific features - Length of the text in characters, number of words, number of lines per document, average number of words per line, number of references etc. Research papers often have more lines per page than theses which have more lines per page than slides and thus can be useful in classifying the document.
3. Section specific features - Section names and their positions in the documents to capture intuition about the structure of the document. Example: in a paper, the acknowledgements section is usually at the end whereas in a thesis it is at the beginning
4. Containment features - Containment if spefific words/phrases in the document like "this paper", "this book", "this thesis", "research interests". The intuition is that the author will use "in this paper" or similar terms to describe what the document is about

What is the conclusion of the paper? Do they compare between classifiers, between feature sets, or on some other dimension? 
Answer: The paper compares between different feature sets. It compares the perforance of classifiers trained using the novel structural fetures proposed by the authors as well as the bag of words model vis a vis classifiers trained using only the bag of words model. The classifier trained using only bag of words achieves an accuracy of 72.6% whereas the classifier with the structural features achieves an accuracy of 89.1%. This can be explained by the fact that the bag of words model contains significant noise. Structural features in turn are features that are relevant for the document classification task at hand.

Give a one-sentence summary of the message that they are trying to leave for the reader.
Answer: Using specific features tailored to the document and which harness information relevant for the clasdification task will result in much better classification accuracy than the bag of words model given the same classifiers.
