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
twenty_train, twenty_test = [
    fetch_20newsgroups(subset=subset, categories=categories,
        shuffle=True, random_state=42)
    for subset in ['train', 'test']
]

train_data = twenty_train.data
train_target = twenty_train.target
train_target_names = twenty_train.target_names

test_data = twenty_test.data
test_target = twenty_test.target
test_target_names = twenty_test.target_names


# --- build fq data from text
count_vect_t = CountVectorizer()
count_vect_t.fit(train_data)
train_data_count_vect = count_vect_t.transform(train_data)


# --- transform fq to TF-IDF
tfidf_t = TfidfTransformer(use_idf=False)  # can be True XXX
tfidf_t.fit(train_data_count_vect)
train_data_tfidf = tfidf_t.transform(train_data_count_vect)


# --- train a MultinomialNB classifier -- an estimator ("_e")
multinb_e = MultinomialNB()
multinb_e.fit(train_data_tfidf, train_target)


# --- predict values for new data -- just 2 documents
test_mini = ['God is love', 'OpenGL on the GPU is fast']
test_mini_count_vect = count_vect_t.transform(test_mini)
test_mini_tfidf = tfidf_t.transform(test_mini_count_vect)

predicted = multinb_e.predict(test_mini_tfidf)

for doc, category in zip(test_mini, predicted):
    print('%r => %s' % (doc, train_target_names[category]))


# --- same as the above, together as a Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(train_data, train_target)


# --- evaluation
predicted = text_clf.predict(test_data)
print('MultinomialNB', np.mean(predicted == test_target))


# --- detailed performance analysis
print(metrics.classification_report(test_target, predicted,
    target_names=test_target_names))
print(metrics.confusion_matrix(test_target, predicted))


# --- can we do better using an SVM?
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])
text_clf.fit(train_data, train_target)


# --- evaluation
predicted = text_clf.predict(test_data)
print('SGDClassifier', np.mean(predicted == test_target))


# --- detailed performance analysis
print(metrics.classification_report(test_target, predicted,
    target_names=test_target_names))
print(metrics.confusion_matrix(test_target, predicted))


# --- parameter tuning using grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-1, 1e-2, 1e-3, 1e-4),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(train_data, train_target)

print('GridSearchCV', gs_clf.best_score_)
print(gs_clf.best_params_)


# --- take at look at the best parameter setting provided by GridSearchCV
# XXX izé, ez miért is nem ua eredményt adja,
# XXX mint a gs_clf.best_score_ ???
#     => arra számítottam volna erősen! :)
text_clf = Pipeline([
    ('vect', CountVectorizer(
        ngram_range=gs_clf.best_params_['vect__ngram_range'])),
    ('tfidf', TfidfTransformer(
        use_idf=gs_clf.best_params_['tfidf__use_idf'])),
    ('clf', SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=gs_clf.best_params_['clf__alpha'],
        random_state=42,
        max_iter=5,
        tol=None
    )),
])
text_clf.fit(train_data, train_target)


# --- evaluation
predicted = text_clf.predict(test_data)
print('SGDClassifier/best', np.mean(predicted == test_target))


# --- detailed performance analysis
print(metrics.classification_report(test_target, predicted,
    target_names=test_target_names))
print(metrics.confusion_matrix(test_target, predicted))

