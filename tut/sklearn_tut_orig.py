"""
This module contains code explained
in the "Working With Text Data" scikit-learn tutorial
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
using the "Twenty Newsgroups" dataset.
"""

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

import numpy as np

categories = [
    'alt.atheism',
    'soc.religion.christian',
    'comp.graphics',
    'sci.med'
]


# --- load text data
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)

#print(twenty_train.target_names)
#print(len(twenty_train.data))
#print(len(twenty_train.filenames))

#print("\n".join(twenty_train.data[0].split("\n")[:3]))
#print(twenty_train.target_names[twenty_train.target[0]])

#print(twenty_train.target[:10])

#for t in twenty_train.target[:10]:
#    print(twenty_train.target_names[t])


# --- build fq data from text
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

#print(X_train_counts.shape)
#print(count_vect.vocabulary_.get(u'algorithm'))


# --- transform fq to TF-IDF
tf_transformer = TfidfTransformer(use_idf=False)  # can be True XXX
tf_transformer.fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

#print(X_train_tf.shape)


# --- train a MultinomialNB classifier
clf = MultinomialNB()
clf.fit(X_train_tf, twenty_train.target)


# --- predict values for new data
docs_new = ['God is love', 'OpenGL on the GPU is fast']

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))


# --- same as the above, together as a Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(twenty_train.data, twenty_train.target)


# --- evaluation
predicted = text_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))


# --- can we do better using an SVM?
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))


# --- detailed performance analysis
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))


# --- parameter tuning uing grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)], # XXX ???
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-1, 1e-2, 1e-3, 1e-4),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
#print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

print(gs_clf.best_score_)
print(gs_clf.best_params_)
#print(gs_clf.cv_results_)

