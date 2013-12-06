#%% Imports
import numpy as np
from scipy.misc import imrotate
from scipy.ndimage import convolve
from PyWiseRF import WiseRF

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
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
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

train = np.genfromtxt('train.csv', delimiter=',')[1:]
target = train[:,0]
train = train[:,1:]
test = np.genfromtxt('test.csv', delimiter=',')[1:]

#%% Rotates and nudges dataset, trains predictor
ntrain,ntarget = rotate_dataset(train,target)
ntrain,ntarget = nudge_dataset(ntrain,ntarget)

wtrees = WiseRF(n_jobs=-1,n_estimators=512) 
wtrees.fit(ntrain,ntarget)
wtrees.score(ntrain,ntarget)
result_svm_rbm = wtrees.predict(test)

#%%

f=open('result_wtrees.csv','w')
f.write('ImageId,Label\n')

count=1

for x in result_svm_rbm:
    f.write('%d,%d\n' % (count,x))
    count += 1

f.close()