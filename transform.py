# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:09:00 2013

@author: kterao
"""

%pylab inline
os.chdir('/home/kterao/git/kaggle-digit/')

data_train = genfromtxt('train.csv', dtype=int, delimiter=',', skip_header=1)

train = MLData(data_train[:1,0], data_train[:1,1:])
X = train.feature
y = train.target
print 'num samples', train.n


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

X = add_image_meta(X)
print X[:,-4:]
# Symmetry: across X, across Y, across X on right/left, across Y on top/bottom
# average intensity: all, right, left, top, bottom, 

X2, y2 = nudge2_dataset(X, y)

