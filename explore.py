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

data_train = genfromtxt('C:/Users/Ken/Projects/kaggle-digit/train.csv', 
    dtype=int, delimiter=',', skip_header=1)

#data_test = genfromtxt('C:/Users/Ken/Projects/kaggle-digit/test.csv', 
#    dtype=int, delimiter=',', skip_header=1)

class MLData:
    def __init__(self, target, feature):
        self.target = target
        self.feature = feature
        self.n = self.feature.shape[0]
        self.k = self.feature.shape[1]
        self.image = [self.feature[i].reshape(28,28) 
            for i in arange(self.n)]

def imageView(target, feature):
    image = [feature[i].reshape(28,28) for i in arange(feature.shape[0])]
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(image[i], cmap=plt.cm.binary, interpolation='nearest')
        
        # label the image with the target value
        ax.text(0, 7, str(target[i]))

## Data set
train = MLData(data_train[:,0], data_train[:,1:])
imageView(train.target, train.feature)


X_train, X_val, y_train, y_val = cross_validation.train_test_split(
    train.feature, train.target, test_size=0.4)

## Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_hat = gnb.fit(X_train, y_train).predict(X_val)
1. * (y_hat != y_val).sum() / y_val.shape[0]

imageView(y_hat, X_val)

## To try:
## Naive Bayes
## SVM
## Unsupervised PCA feature extraction
## Plots to test differentiation of new factors
## Cross validation methods
## Eigen images
## detect right/left handed?
## pct of pixels "dark", pct pixels "light"

# Try new factors
# Symmetry: across X, across Y, across X on right/left, across Y on top/bottom
# average intensity: all, right, left, top, bottom, 
# average of columns and rows
# 


