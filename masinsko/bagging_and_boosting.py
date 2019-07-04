#algoritmi koji spadaju u grupe bagging i boosting algoritama
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold

from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn import tree


#boosting
def ada_boost_implement(X, y):
    #implementacija ada boost algoritma
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    classifier = AdaBoostClassifier()
    params = {
        'n_estimators': [500, 1000, 2000],
        'learning_rate' : [0.001, 0.01, 0.1, 1]
    }
    '''
    gs = GridSearchCV(estimator=classifier, param_grid=params, scoring='f1_micro', n_jobs=1, cv=crossvalidation)
    gs.fit(X, y)
    print(gs.best_params_)
    print(gs.best_score_)
    '''
    # learning rate = 0.01, n_estimators = 500
    classifier = AdaBoostClassifier(learning_rate=0.01, n_estimators=500)
    classifier.fit(X, y)
    return classifier

#bagging


def random_forest_implement(X, y):
    classifier = RandomForestClassifier()

    grid_param = {
        'n_estimators' : [100, 300, 500, 800, 1000],
        'criterion' : ['gini', 'entropy'],
        'bootstrap' : [True, False]
    }
    '''
    gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='f1_micro', cv=5, n_jobs=-1)
    gd_sr.fit(X, y)
    print(gd_sr.best_params_)
    print(gd_sr.best_estimator_)
    #{'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 500}
   # classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=True)
   # classifier.fit(X, y)
  #  return classifier
    '''
    classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=500,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
    classifier.fit(X, y)
    return  classifier

def bagging_implement_dtc(X, y):
    classifier = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
    classifier.fit(X, y)
    return  classifier

def bagging_implement_etc(X, y):
    classifier = BaggingClassifier(tree.ExtraTreeClassifier(random_state=1))
    classifier.fit(X, y)
    return  classifier

def extraTree_implement(X,y):
    classifier = ExtraTreesClassifier()
    params = {
        'n_estimators' : [100, 300, 500, 800, 1000],
        'criterion' : ['gini', 'entropy'],
        'bootstrap' : [True, False]
    }
    '''
    gs = GridSearchCV(estimator=classifier, param_grid=params, scoring='f1_micro', cv = 5)
    gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_estimator_)
    '''
    classifier = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                     max_depth=None, max_features='auto', max_leaf_nodes=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=1000,
                     n_jobs=None, oob_score=False, random_state=None, verbose=0,
                     warm_start=False)
    classifier.fit(X,y)
    return classifier
