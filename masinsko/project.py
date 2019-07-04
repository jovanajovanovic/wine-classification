import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  AdaBoostClassifier,ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.decomposition import PCA
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib

import bagging_and_boosting
import algorithm

def load_data(path):
    data = pd.read_csv(path)

    data = pd.DataFrame(data)

    X = data.drop('quality', axis = 1)
    y = data['quality']

    #20% skupa ide za test skup, a na 80% skupa cemo trenirati algoritme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data("wine.csv")

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)


    '''
    pca = PCA().fit(X_train)
    #za odabir broja komponenti u pca
    print(matplotlib.get_backend())
    matplotlib.rcParams["backend"] = "TkAgg"
    plt.switch_backend("TkAgg")
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Num of components')
    plt.ylabel('Variance (%)')
    plt.title('Wine Dataser Explained Variance')
    plt.show()
    '''
    pca = PCA(n_components=9)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    #jedan po jedan algoritam cemo pozivati

    #svm algoritam
    print("result: ")

    svm_classifier = algorithm.svm_implement(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    f1_svm = f1_score(y_test, y_pred, average='micro')
    print("==================================")
    print("svm f1: ", f1_svm)
    print("==================================")

    #ada boost
    ada_boost = bagging_and_boosting.ada_boost_implement(X_train, y_train)
    y_pred = ada_boost.predict(X_test)
    f1_ada = f1_score(y_test, y_pred, average='micro')
    print("ada boost: ", f1_ada)
    print("==================================")


    et = bagging_and_boosting.extraTree_implement(X_train, y_train)
    y_pred = et.predict(X_test)
    f1_rf = f1_score(y_test, y_pred, average='micro')
    print("extra trees: ", f1_rf)
    print("==================================")



    rf = bagging_and_boosting.random_forest_implement(X_train, y_train)
    y_pred = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred, average='micro')
    print("random forest: ", f1_rf)
    print("==================================")


    bgd = bagging_and_boosting.bagging_implement_dtc(X_train, y_train)
    y_pred = bgd.predict(X_test)
    bg_f1 = f1_score(y_test, y_pred, average='micro')
    print("bagging with dtc: ", bg_f1)
    print("==================================")

    bge = bagging_and_boosting.bagging_implement_etc(X_train, y_train)
    y_pred = bge.predict(X_test)
    bg_f1 = f1_score(y_test, y_pred, average='micro')
    print("bagging with etc: ", bg_f1)
    print("==================================")


    knn = algorithm.knn_model(X_train, y_train)
    y_predict = knn.predict(X_test)
    print("knn: ", f1_score(y_test, y_predict, average='micro'))
    print("==================================")


    voting = algorithm.voting_implement(X_train, y_train)
    y_predict = voting.predict(X_test)
    print("voting: ", f1_score(y_test, y_predict, average='micro'))
    print("==================================")
