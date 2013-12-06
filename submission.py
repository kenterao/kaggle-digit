# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:46:56 2013

@author: kterao
"""

import os
import sys

from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import cross_validation

%pylab inline
os.chdir('/home/kterao/git/kaggle-digit/')

data_train = genfromtxt('train.csv', dtype=int, delimiter=',', skip_header=1)
data_test = genfromtxt('test.csv', dtype=int, delimiter=',', skip_header=1)

## Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=256, max_depth=None, 
                           min_samples_split=10, n_jobs=-1, 
                           min_samples_leaf=5)

X_train = data_train[:,1:]
y_train = data_train[:,0]
X_test = data_test[:,:]

## Transform data only on the test set, not the validation set
X_trans, y_trans = nudge2_dataset(X_train, y_train)

## Add metadata
X_trans = add_image_meta(X_trans)
X_test_trans = add_image_meta(X_test)

clf.fit(X_trans, y_trans)
y_test = clf.predict(X_test_trans)

imageView(X_test, y_test)

f=open('result2.csv','w')
f.write('ImageId,Label\n')

count=1

for y in y_test:
    f.write('%d,%d\n' % (count,y))
    count += 1

f.close()










