# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:26:24 2013

@author: kterao
"""

import os
import sys

from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import cross_validation
import numpy as np
from scipy.misc import imrotate
from scipy.ndimage import convolve



class MLData:
    def __init__(self, target, feature):
        self.target = target
        self.feature = feature
        self.n = self.feature.shape[0]
        self.k = self.feature.shape[1]
        self.image = [self.feature[i].reshape(28,28) 
            for i in arange(self.n)]

def imageView(feature, target):
    image = [feature[i].reshape(28,28) for i in arange(feature.shape[0])]
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, 
                        wspace=0.05)                        
    # plot the digits: each image is 28x28 pixels
    for i in range(min(64,target.shape[0])):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(image[i], cmap=plt.cm.binary, interpolation='nearest')
        
        # label the image with the target value
        ax.text(0, 7, str(target[i]))

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 28x28 images in X around by 1px to left, right, down, up
    
    Try half -pixel nudges by averaging intensity
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 1],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [1, 0, 0]],

        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]]

    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(1+len(direction_vectors))], axis=0)
    return X, Y

def nudge2_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 28x28 images in X around by 1px to left, right, down, up
    
    Try half -pixel nudges by averaging intensity
    """
    direction_vectors = [
        [[0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 1],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [1, 0, 0]],

        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]]

    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(1+len(direction_vectors))], axis=0)
    return X, Y
    
def rotate_dataset(X, Y):
    """
    This produces a dataset 2 times bigger than the original one,
    by rptating the 28x28 images in 10 degrees clockwise and counter clockwise
    """
    angles = [-10,10]

    
    rotate = lambda x, w: imrotate(x.reshape((28, 28)), w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(rotate, 1, X, angle)
                        for angle in angles])
    Y = np.concatenate([Y for _ in range(3)], axis=0)
    return X, Y

## Average pixel intensity
def avg_pixel(image):
    var = [ image[i].mean() for i in arange(len(image)) ]
    return np.array(var)[...,None]

## Count empty pixels
def white_count(image):
    var = [ (image[i].reshape(-1)<5).sum() for i in arange(len(image)) ]
    return np.array(var)[...,None]

## Count dark pixels
def black_count(image):
    var = [ (image[i].reshape(-1)>250).sum() for i in arange(len(image)) ]
    return np.array(var)[...,None]    

## Axis 0 average
def ax0_avg(image):
    var = [ mean(image[i],axis=0) for i in arange(len(image)) ]
    return np.array(var)
    
## Axis 1 average
def ax1_avg(image):
    var = [ mean(image[i],axis=1) for i in arange(len(image)) ]
    return np.array(var)

## Axis 0 symmetry, x-direction symmetery, about y-axis
def ax0_sym(image, perm):
    var = [ (np.abs(np.dot(image[i], perm))).mean() \
        for i in arange(len(image)) ]
    return np.array(var)[...,None]

## Axis 1 symmetry
def ax1_sym(image, perm):
    var = [ (np.abs(np.dot(perm, image[i]))).mean() \
        for i in arange(len(image)) ]
    return np.array(var)[...,None]    

## top symmetry
def top_sym(image, perm):
    var = [ (np.abs(np.dot(image[i], perm)))[:14,:].mean() \
        for i in arange(len(image)) ]
    return np.array(var)[...,None]

## bottom symmetry
def bottom_sym(image, perm):
    var = [ (np.abs(np.dot(image[i], perm)))[-14:,:].mean() \
        for i in arange(len(image)) ]
    return np.array(var)[...,None]

## left symmetry
def left_sym(image, perm):
    var = [ (np.abs(np.dot(perm, image[i])))[:,:14].mean() \
        for i in arange(len(image)) ]
    return np.array(var)[...,None]    

## left symmetry
def right_sym(image, perm):
    var = [ (np.abs(np.dot(perm, image[i])))[:,-14:].mean() \
        for i in arange(len(image)) ]
    return np.array(var)[...,None]    

def add_image_meta(X):
    perm = np.eye(28)
    for ii in arange(28):
        perm[ii,28-ii-1] = -1
        
    image = [ X[i].reshape(28,28) for i in arange(X.shape[0]) ]
    X = np.concatenate((X, avg_pixel(image)), axis=1)
    X = np.concatenate((X, white_count(image)), axis=1)
    X = np.concatenate((X, black_count(image)), axis=1)
    X = np.concatenate((X, ax0_avg(image)), axis=1)
    X = np.concatenate((X, ax1_avg(image)), axis=1)
    X = np.concatenate((X, ax0_sym(image, perm)), axis=1)
    X = np.concatenate((X, ax1_sym(image, perm)), axis=1)
    X = np.concatenate((X, top_sym(image, perm)), axis=1)
    X = np.concatenate((X, bottom_sym(image, perm)), axis=1)
    X = np.concatenate((X, left_sym(image, perm)), axis=1)
    X = np.concatenate((X, right_sym(image, perm)), axis=1)
    return X            