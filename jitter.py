# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:01:50 2013

@author: kterao
"""
from scipy.ndimage import convolve

%pylab inline
os.chdir('/home/kterao/git/kaggle-digit/')

data_train = genfromtxt('train.csv', dtype=int, delimiter=',', skip_header=1)

train = MLData(data_train[:15000,0], data_train[:15000,1:])
X = train.feature
y = train.target
print 'num samples', train.n

#X = X[0:1]
#y = y[0:1]
#X, y = nudge2_dataset(X, y)
#imageView(X, y)
       
## 5-fold CV
from sklearn.cross_validation import cross_val_score
kf = cross_validation.KFold(train.n, n_folds=3)

## Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=256, max_depth=None, 
                           min_samples_split=50, n_jobs=-1, 
                           min_samples_leaf=25)

def nojitter(X, y):
    return X, y
    
def nometa(X):
    return X    

def digit_run(clf, kf, jitter=nojitter, add_meta=nometa):
    ## Cross Validation Scores
    scores = np.zeros(kf.n_folds)    
    for k_idx, (train_idx, val_idx) in enumerate(kf):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        ## Transform data only on the test set, not the validation set
        X_trans, y_trans = jitter(X_train, y_train)
        
        ## Add metadata
        X_trans = add_meta(X_trans)
        X_val = add_meta(X_val)
        
        clf.fit(X_trans, y_trans)
        y_hat = clf.predict(X_val)
        score = 1. * (y_hat == y_val).sum() / y_val.shape[0]
        scores[k_idx] = score
    return scores



scores = digit_run(clf, kf, nojitter, add_image_meta)
print scores.mean(), scores
# n_folds = 5, n_estimators = 200, ExtraTrees, nojitter
# score 0.930

scores = digit_run(clf, kf, nudge_dataset, add_image_meta)
print scores.mean(), scores
# n_folds = 5, n_estimators = 200, ExtraTrees, nudge 4 dir
# score

scores = digit_run(clf, kf, nudge2_dataset, add_image_meta)
print scores.mean(), scores
# n_folds = 5, n_estimators = 200, ExtraTrees, nudge 4 dir
# score

def rotate_and_nudge(X, y):
    X2, y2 = rotate_dataset(X, y)
    X3, y3 = nudge2_dataset(X2, y2)
    return X3, y3
scores = digit_run(clf, kf, rotate_and_nudge, add_image_meta)
print scores.mean(), scores
# n_folds = 5, n_estimators = 200, ExtraTrees, nudge 4 dir
# score 0.964


    