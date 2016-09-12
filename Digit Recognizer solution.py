# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 16:08:46 2016

@author: Trace
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.ix[:, 1:785]
Y = train.ix[:, 0]

X_test = test

ImageID = test.index.values + 1

neighbors = KNeighborsClassifier(n_neighbors=3)
neighbors.fit(X, Y)
Y_test_neighbors = neighbors.predict(X_test)
print(neighbors.score(X, Y))

'''
clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit(X, Y)
Y_test_dt = clf_dt.predict(X_test)
print(clf_dt.score(X, Y))
'''

'''
clf_rf = ske.RandomForestClassifier(n_estimators=50)
clf_rf.fit(X,Y)
Y_test_rf = clf_rf.predict(X_test)
print(clf_rf.score(X, Y))
'''

'''
clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
clf_gb.fit(X,Y)
Y_test_gb = clf_gb.predict(X_test)
print(clf_gb.score(X, Y))
'''

'''
clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_rf = ske.RandomForestClassifier(n_estimators=50)
eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf)])
eclf.fit(X,Y)
Y_test_eclf = eclf.predict(X_test)
print(eclf.score(X, Y))
'''

'''
submission = pd.DataFrame({'ImageID': ImageID, 'Label': Y_test_dt})
submission.to_csv('clf_dt_digit_rec.csv', index=False)
'''

'''
submission = pd.DataFrame({'ImageID': ImageID, 'Label': Y_test_rf})
submission.to_csv('clf_rf_digit_rec.csv', index=False)
'''
'''
submission = pd.DataFrame({'ImageID': ImageID, 'Label': Y_test_gb})
submission.to_csv('clf_gb_digit_rec.csv', index=False)
'''

'''
submission = pd.DataFrame({'ImageID': ImageID, 'Label': Y_test_eclf})
submission.to_csv('clf_eclf_digit_rec.csv', index=False)
'''

submission = pd.DataFrame({'ImageID': ImageID, 'Label': Y_test_neighbors})
submission.to_csv('neighbors_digit_rec.csv', index=False)