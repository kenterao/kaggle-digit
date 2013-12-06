# -*- coding: utf-8 -*-
"""
Created on Tue Dec 03 21:37:39 2013

@author: Ken
"""
import os
import sys

from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import cross_validation

%pylab inline

# Home windows machine
#filename = 'C:/Users/Ken/Projects/kaggle-digit/train.csv'
# Work Fedora machine
filename = '/home/kterao/git/kaggle-digit/train.csv'

data_train = genfromtxt(filename, dtype=int, delimiter=',', skip_header=1)

#data_test = genfromtxt('C:/Users/Ken/Projects/kaggle-digit/test.csv', 
#    dtype=int, delimiter=',', skip_header=1)

## Data set
## Truncate data set for testing
train = MLData(data_train[:,0], data_train[:,1:])
X = train.feature
y = train.target
print 'num samples', train.n
imageView(X, y)

## Create jitter function





## 5-fold CV
from sklearn.cross_validation import cross_val_score
kf = cross_validation.KFold(train.n, n_folds=5)

## Naive Bayes, 5-fold CV
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
scores = cross_val_score(clf, X, y, cv=kf)
print clf.__str__
print scores.mean(), scores, '\n'
####

## Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1)
scores = cross_val_score(clf, X, y, cv=kf)
print clf.__str__
print scores.mean(), scores, '\n'
####

## Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             min_samples_split=1)
scores = cross_val_score(clf, X, y, cv=kf)
print clf.__str__
print scores.mean(), scores, '\n'
####

## Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=200, max_depth=None,
                           min_samples_split=1)
scores = cross_val_score(clf, X, y, cv=kf)
print clf.__str__
print scores.mean(), scores, '\n'
####

                       

## To try:
## Naive Bayes
## SVM
## Unsupervised PCA feature extraction
## Eigen images
## Cross validation methods
## KNN, cross validation on K
## Deep Belief Nets
## Random Forest
## Dropout
## Maxout

# Try new factors
## Plots to test differentiation of new factors
# Symmetry: across X, across Y, across X on right/left, across Y on top/bottom
# average intensity: all, right, left, top, bottom, 
# average of columns and rows
# detect right/left handed?
# pct of pixels "dark", pct pixels "light"


