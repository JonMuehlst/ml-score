from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.cross_validation import    StratifiedShuffleSplit, ShuffleSplit, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import   LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import load_svmlight_files

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd
import numpy as np
import scipy.stats as stats

import os
import logging
from timeit import default_timer as timer

from xgboost import XGBClassifier

from CategoryClassifier import CategoryClassifier
from new_clf.emb_svm import emb_svm
from translate_labels import transform_labels

"""
Changes:
0. Docstrings and documetation
1. Use python logger
"""




def loadData(size):
    # See ShuffleSplit below
    # Load data
    data_path = os.getcwd() + "/data_set/libSVM"

    train_0 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_0_train"
    test_0 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_0_test"
    train_1 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_1_train"
    test_1 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_1_test"
    train_2 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_2_train"
    test_2 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_2_test"
    train_3 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_3_train"
    test_3 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_3_test"
    train_4 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_4_train"
    test_4 = data_path + "/samples_25.2.16_comb_triple.csv_libSVM_4_test"

    X_train_0, y_train_0, X_test_0, y_test_0 = load_svmlight_files(
    (train_0, test_0))
    X_train_1, y_train_1, X_test_1, y_test_1 = load_svmlight_files(
    (train_1, test_1))
    X_train_2, y_train_2, X_test_2, y_test_2 = load_svmlight_files(
    (train_2, test_2))
    X_train_3, y_train_3, X_test_3, y_test_3 = load_svmlight_files(
    (train_3, test_3))
    X_train_4, y_train_4, X_test_4, y_test_4 = load_svmlight_files(
    (train_4, test_4))

    # Set dataset size and appropriate indexed in a stratified manner
    sss_train_0 = ShuffleSplit(len(y_train_0), n_iter=1, train_size=size,
                                                random_state=0)
    sss_train_1 = ShuffleSplit(len(y_train_1), n_iter=1, train_size=size,
                                                random_state=0)
    sss_train_2 = ShuffleSplit(len(y_train_2), n_iter=1, train_size=size,
                                                random_state=0)
    sss_train_3 = ShuffleSplit(len(y_train_3), n_iter=1, train_size=size,
                                                random_state=0)
    sss_train_4 = ShuffleSplit(len(y_train_4), n_iter=1, train_size=size,
                                                random_state=0)

    for train_ind,_ in sss_train_0:
        df_train_0 = pd.DataFrame(X_train_0.toarray())
        df_train_0 = df_train_0.loc[train_ind]
        y_train_0 = y_train_0[train_ind]

    df_test_0 = pd.DataFrame(X_test_0.toarray())


    for train_ind,_ in sss_train_1:
        df_train_1 = pd.DataFrame(X_train_1.toarray())
        df_train_1 = df_train_1.loc[train_ind]
        y_train_1 = y_train_1[train_ind]

    df_test_1 = pd.DataFrame(X_test_1.toarray())


    for train_ind,_ in sss_train_2:
        df_train_2 = pd.DataFrame(X_train_2.toarray())
        df_train_2 = df_train_2.loc[train_ind]
        y_train_2 = y_train_2[train_ind]

    df_test_2 = pd.DataFrame(X_test_2.toarray())


    for train_ind,_ in sss_train_3:
        df_train_3 = pd.DataFrame(X_train_3.toarray())
        df_train_3 = df_train_3.loc[train_ind]
        y_train_3 = y_train_3[train_ind]

    df_test_3 = pd.DataFrame(X_test_3.toarray())

    for train_ind,_ in sss_train_4:
        df_train_4 = pd.DataFrame(X_train_4.toarray())
        df_train_4 = df_train_4.loc[train_ind]
        y_train_4 = y_train_4[train_ind]

    df_test_4 = pd.DataFrame(X_test_4.toarray())


    X_train_0 = df_train_0
    X_test_0 = df_test_0
    X_train_1 = df_train_1
    X_test_1 = df_test_1
    X_train_2 = df_train_2
    X_test_2 = df_test_2
    X_train_3 = df_train_3
    X_test_3 = df_test_3
    X_train_4 = df_train_4
    X_test_4 = df_test_4

    return [(X_train_0, y_train_0, X_test_0, y_test_0),
            (X_train_1, y_train_1, X_test_1, y_test_1),
            (X_train_2, y_train_2, X_test_2, y_test_2),
            (X_train_3, y_train_3, X_test_3, y_test_3),
            (X_train_4, y_train_4, X_test_4, y_test_4)]

def getFeatureGroups():
    # Feature groups
    # couples of (Group name, columns to drop)
    group_names = [
                    'Combined features', 'Common only',
                    'New only', 'Combined no peaks',
                    'Combined no SSL', 'Combined no TCP',
                    'Peaks only', 'Stats only',
                    'Common stats only'
                  ]
    columns_to_drop = [
                        [], range(13) + range(23,41) + [68],
                        range(13,23) + range(41,68),
                        range(23,41), [0,1,2] + [6,7,8,9,10,11,12],
                        range(3,6) + range(66,69), range(23) + range(41,69),
                        range(13) + range(66,69),
                        range(13) + range(23,41) + range(66,69)
                      ]

    return group_names, columns_to_drop

def getEnsemble(voting='hard'):
    # Prepare ensemble method
    estimators = []
    model1 = KNeighborsClassifier(n_neighbors=16,algorithm='ball_tree',
                                    metric='canberra', n_jobs=-1)
    estimators.append(('knn', model1))
    model2 = SVC(gamma=0.0078125,C=8192, probability=False)
    estimators.append(('svmrbf', model2))
    model3 = DecisionTreeClassifier()#max_depth=50)
    estimators.append(('DecisionTree', model3))
    model4 = RandomForestClassifier(n_estimators=100, oob_score=True,
                                    n_jobs=-1)
    estimators.append(('RandomForest', model4))
    model5 = XGBClassifier(max_depth=10, n_estimators=100, learning_rate=0.1)
    estimators.append(('XGBoost', model5))

    ensemble = VotingClassifier(estimators,voting=voting)
    return ensemble

def getClassifiers():
    estimators = []

    knn_params=[{
    'n_neighbors':[4,6,8,10,12,14,16,18,20],
    'n_jobs':[-1],
    'weights':['uniform','distance'],
    'algorithm':['ball_tree'],
    'metric':['euclidean','manhattan','chebyshev','hamming','canberra']
    }]
    knn = KNeighborsClassifier()
    model1 = GridSearchCV(knn, knn_params, n_jobs=-1)
    estimators.append(('knn', model1))

    svc_params = [{
    'gamma':[2**-15, 2**-13 , 2**-11 , 2**-9 , 2**-7 , 2**-5 , 2**-3 , 2**-1 , 2**1 , 2**3],
    'C':[2**-5, 2**-3, 2**-1 , 2**1 , 2**3 , 2**5 , 2**7 , 2**9 , 2**11 , 2**13 , 2**15],
    'probability':[False]
    }]
    svmrbf = SVC()
    model2 = GridSearchCV(svmrbf, svc_params, n_jobs=-1)
    estimators.append(('svmrbf', model2))

    # model3 = DecisionTreeClassifier()#max_depth=50)
    # estimators.append(('DecisionTree', model3))

    rf_params = [{'n_estimators':[20,40,60,80,100,120],
                    'oob_score':[True],
                    'n_jobs':[-1]}]
    rf = RandomForestClassifier()
    model4 = GridSearchCV(rf, rf_params, n_jobs=-1)
    estimators.append(('RandomForest', model4))

    # xgb_params = [{'max_depth':[5,10,15,20], 'n_estimators':[33,66,99], 'learning_rate':[0.001,0.1,0.5]}]
    # xgb_clf = XGBClassifier()
    # model5 = GridSearchCV(xgb_clf, xgb_params, n_jobs=-1)
    # estimators.append(('XGBoost', model5))

    # model6 = CategoryClassifier()
    # estimators.append(('CategoryClassifierXGBoost', model6))
    # model7 = emb_svm(C=8192, distance='canberra', threshold_ind=9)
    # estimators.append(('embeddingSVM', model7))

    # names = ['knn', 'svmrbf', 'DecisionTree', 'RandomForest', 'XGBoost', 'CategoryClassifierXGBoost', 'embeddingSVM']
    names = ['knn', 'svmrbf', 'RandomForest']
    # names = ['embeddingSVM']
    return names, estimators

def processData(X_train, X_test, features_to_remove):
    return X_train.drop(features_to_remove, axis=1), X_test.drop(features_to_remove, axis=1)

def stat_test(res1 , res2, equal_var=False):
    return stats.ttest_ind(a=res1,b=res2,equal_var=equal_var)

def pairwise_ttest(acc_dict):
    dict_keys = acc_dict.keys()
    ind_keys = [tuple(('L_' + s1,s2)) for s1,s2 in dict_keys]
    col_keys = [tuple(('R_' + s1,s2)) for s1,s2 in dict_keys]
    df_m = pd.DataFrame(np.zeros((len(acc_dict), len(acc_dict))),
                        index=ind_keys, columns=col_keys)
    df_p = pd.DataFrame(np.zeros((len(acc_dict), len(acc_dict))),
                        index=ind_keys, columns=col_keys)

    for (key11, key12), val1 in acc_dict.iteritems():
        for (key21, key22), val2 in acc_dict.iteritems():
             m, p = stat_test(val1 , val2, equal_var=False)
             df_m.loc[('L_'+key11,key12),('R_'+key21,key22)] = m
             df_p.loc[('L_'+key11,key12),('R_'+key21,key22)] = p

    return df_m, df_p


class FiveFoldRuntimeRecord(object):

    def __init__(self, clf_names,feature_group_names):
        super(FiveFoldRuntimeRecord, self).__init__()

        self.res_df = []
        self.results = []
        self.train_times = []
        self.test_times = []
        self.accuracy_dict = {}

        self.fold_names = ['fold_0','fold_1','fold_2','fold_3','fold_4', 'avg']

        indexes = pd.MultiIndex.from_product([clf_names,feature_group_names,self.fold_names],
                                            names=['clf_name','feature_group','fold_num'])

        table_size = (len(self.fold_names))*(len(clf_names))*(len(feature_group_names))
        column_names = ['clf_params',
                                'features_removed',
                                'avg_train_time',
                                'avg_test_time',
                                'avg_score',
                                'min_score',
                                'max_score']

        self.res_df = pd.DataFrame( np.zeros((table_size, len(column_names))),
                                    index=indexes, columns=column_names)


    def updateFold(self, fold_i, clf, score, clf_name, fg_name, features_to_drop):
        self.results.append(score)

        # Assuming gridsearchCV
        self.res_df.loc[(clf_name,fg_name,self.fold_names[fold_i]),'clf_params'] = str(repr(clf.best_estimator_.get_params(deep=False)))
        self.res_df.loc[(clf_name,fg_name,self.fold_names[fold_i]),'features_removed'] = str(features_to_drop)
        self.res_df.loc[(clf_name,fg_name,self.fold_names[fold_i]),'avg_score'] = score

    def updateFoldTimes(self, train_time, test_time):
        self.train_times.append(train_time)
        self.test_times.append(test_time)

    def updateFiveFoldSummary(self, clf_name, fg_name, features_to_drop):
        self.res_df.loc[(clf_name,fg_name,'avg'),'avg_train_time'] = np.average(self.train_times)
        self.res_df.loc[(clf_name,fg_name,'avg'),'avg_test_time'] = np.average(self.test_times)
        self.res_df.loc[(clf_name,fg_name,'avg'),'features_removed'] = str(features_to_drop)
        self.res_df.loc[(clf_name,fg_name,'avg'),'avg_score'] = np.average(self.results)
        self.res_df.loc[(clf_name,fg_name,'avg'),'min_score'] = np.min(self.results)
        self.res_df.loc[(clf_name,fg_name,'avg'),'max_score'] = np.max(self.results)
        self.accuracy_dict[(clf_name,fg_name)] = self.results
        logger.info('Average: ' + repr(np.average(self.results)))
        logger.info('Min: ' + repr(np.min(self.results)))
        logger.info('Max: ' + repr(np.max(self.results)))
        self.results = []
        self.train_times = []
        self.test_times = []

    def saveSizeResults(self, size):
        ttest_df_m, ttest_df_p = pairwise_ttest(self.accuracy_dict)
        ttest_df_m.to_csv(os.getcwd() + '/' + str(size) +'_ttest_diff_pairs.csv')
        ttest_df_p.to_csv(os.getcwd() + '/' + str(size) +'_ttest_p_val_pairs.csv')
        self.res_df.to_csv(os.getcwd() + '/' + str(size) +'_results.csv')

# Assuming clfs are gridsearchCV
def fiveFold(clf, clf_name, foldsData, fg_name, features_to_drop, ffrr):
    fold_i = 0
    for X_train, y_train, X_test, y_test in foldsData:
        X_train, X_test = processData(X_train, X_test, features_to_drop)
        score = oneFold(X_train, y_train, X_test, y_test, clf, ffrr)

        ffrr.updateFold(fold_i, clf, score, clf_name, fg_name, features_to_drop)
        fold_i = fold_i + 1

def oneFold(X_train,y_train,X_test,y_test,clf, ffrr):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    start_train = timer()
    clf.fit(X_train, y_train)
    end_train = timer()
    train_time = end_train - start_train
    logger.info('Train time: ' + repr(train_time))

    start_test = timer()
    fold_score = clf.score(X_test, y_test)
    logger.info('Fold score: ' + repr(fold_score))
    end_test = timer()
    test_time = end_test - start_test
    logger.info('Test time: ' + repr(test_time))

    ffrr.updateFoldTimes(train_time, test_time)
    return fold_score

def wrapper(size):
    foldsData = loadData(size)
    feature_group_names, feature_groups = getFeatureGroups()
    clf_names, estimators = getClassifiers()

    ffrr = FiveFoldRuntimeRecord(clf_names,feature_group_names)

    for clf_name, clf in estimators:
        logger.info('clf: ' + clf_name)
        logger.info('Params: ' + repr(clf.get_params(deep=False)))

        for fg_name, features_to_drop in zip(feature_group_names,
                                                        feature_groups):
            logger.info('clf: ' + clf_name)
            logger.info('feature group: ' + fg_name)
            logger.info('features removed: ' + repr(features_to_drop))

            fiveFold(clf, clf_name, foldsData, fg_name, features_to_drop, ffrr)
            ffrr.updateFiveFoldSummary(clf_name, fg_name, features_to_drop)

    logger.info('Done with: ' + clf_name)

    ffrr.saveSizeResults(size)


#-----------------------MAIN------------------------------------------------------
if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    # sizes = [50, 100, 200, 300, 400, 500,
            #  600, 700, 800, 900, 1000,2000,5000,10000]
    sizes = [10000]
    # sizes = [570,670]
    for size in sizes:
        wrapper(size)
