import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier, VotingClassifier

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import  LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import bagging_and_boosting

def knn_model(X, y):
    knn = KNeighborsClassifier()
    '''
    params_knn = {'n_neighbors' : np.arange(1,25)}
    knn_gs = GridSearchCV(knn, param_grid=params_knn, cv=5)
    knn_gs.fit(X, y)
    print(knn_gs.best_estimator_)
    print(knn_gs.best_params_)
    '''
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                     weights='uniform')
    knn.fit(X, y)
    return  knn



def log_regression(X, y):
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    return log_reg


def svm_implement(X, y):
    #implementacija svm algoritma sa cross validacijom i izbor najboljih parametara za algoritam
    classifier = svm.SVC()

    param =  {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (0.9,1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
    '''
    gd = GridSearchCV(estimator=classifier, param_grid=param, scoring='f1_micro', cv=5)
    gd.fit(X, y)
    # cv = cross_val_score(gd, X, y, scoring='f1_micro', cv=4)
    print(gd.best_score_)
    print(gd.best_params_)
    #dobijeni su sledeci rezultati 
    {'C': 1, 'decision_function_shape': 'ovo', 'gamma': 1, 'kernel': 'rbf', 'shrinking': True}
    SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=1, kernel='rbf', max_iter=-1,
    probability=False, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
    '''
    classifier = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=1, kernel='rbf', max_iter=-1,
    probability=False, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
    classifier.fit(X, y)
    return classifier



def voting_implement(X, y):
    clf1 = bagging_and_boosting.ada_boost_implement(X,y)
    clf2 = bagging_and_boosting.extraTree_implement(X,y)
    clf3 = svm_implement(X,y)
    clf4 = bagging_and_boosting.random_forest_implement(X,y)
    clf5 = knn_model(X,y)
    clf6 = bagging_and_boosting.bagging_implement_dtc(X,y)
    clf7 = bagging_and_boosting.bagging_implement_etc(X,y)
    estimators = [('knn', clf5),('rf', clf4), ('svm', clf3),('ada', clf1), ('et', clf2), ('bgd', clf6), ('bge', clf7)]

    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X,y)
    return  ensemble





