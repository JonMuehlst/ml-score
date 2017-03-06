import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest

from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_svmlight_files
from sklearn import cross_validation
from sklearn.ensemble import VotingClassifier, BaggingClassifier

from CategoryClassifier import CategoryClassifier
from xgboost import XGBClassifier
from new_clf.emb_svm import emb_svm

import scipy.stats as stats

from timeit import default_timer as timer
from sklearn.cross_validation import ShuffleSplit,KFold

from translate_labels import transform_labels
res_df = []
train_times = []
test_times = []
accuracy_dict = {}


""" mean: 0.99819, std: 0.00051, params:
    {'svmrbf__gamma': 0.0078125, 'knn__algorithm': 'ball_tree',
    'knn__n_neighbors': 16, 'svmrbf__C': 8192, 'knn__metric': 'canberra'} """
def fiveFold(size):

    global train_times
    global test_times
    global accuracy_dict

    cv = loadData(size)
    print 'size: ' + str(size)
    feature_group_names, feature_groups = getFeatureGroups()
    clf_names, estimators = getClassifiers()
    fold_names = ['fold_0','fold_1','fold_2','fold_3','fold_4', 'avg']

    indexes = pd.MultiIndex.from_product([clf_names,feature_group_names,fold_names],
                                        names=['clf_name','feature_group','fold_num'])
    global res_df
    table_size = (len(fold_names))*(len(clf_names))*(len(feature_group_names))
    res_df = pd.DataFrame(
                        np.zeros((table_size, 9)),
                        index=indexes, columns=['clf_params',
                                                'features_removed',
                                                'avg_train_time',
                                                'avg_test_time',
                                                'avg_score',
                                                'min_score',
                                                'max_score',
                                                't-test_RF',
                                                't-test_catXGB'])


    for clf_name, clf in estimators:
        print 'clf: ' + clf_name
        print 'Params: ', repr(clf.get_params(deep=False))
        print

        for fg_name, features_to_drop in zip(feature_group_names,
                                                        feature_groups):

            results = []


            train_times = []
            test_times = []
            print 'clf: ' + clf_name
            print 'feature group: ' + fg_name
            print 'features removed: ', repr(features_to_drop)
            fold_i = 0
            for X_train, y_train, X_test, y_test in cv:
                X_train, X_test = processData(X_train, X_test, features_to_drop)
                print 'Train size: ' + str(len(X_train))
		score = oneFold(X_train, y_train, X_test, y_test, clf)
                results.append(score)

                # Assuming gridsearchCV
                res_df.loc[(clf_name,fg_name,fold_names[fold_i]),'clf_params'] = str(repr(clf.best_estimator_.get_params(deep=False)))
                res_df.loc[(clf_name,fg_name,fold_names[fold_i]),'features_removed'] = str(features_to_drop)

                res_df.loc[(clf_name,fg_name,fold_names[fold_i]),'avg_score'] = score

                fold_i = fold_i + 1

            res_df.loc[(clf_name,fg_name,'avg'),'avg_train_time'] = np.average(train_times)
            res_df.loc[(clf_name,fg_name,'avg'),'avg_test_time'] = np.average(test_times)


            res_df.loc[(clf_name,fg_name,'avg'),'features_removed'] = str(features_to_drop)

            res_df.loc[(clf_name,fg_name,'avg'),'avg_score'] = np.average(results)
            res_df.loc[(clf_name,fg_name,'avg'),'min_score'] = np.min(results)
            res_df.loc[(clf_name,fg_name,'avg'),'max_score'] = np.max(results)
            accuracy_dict[(clf_name,fg_name)] = results
            print 'Average: ', repr(np.average(results))
            print 'Min: ', repr(np.min(results))
            print 'Max: ', repr(np.max(results))
            print 't-test RF: ', repr(stat_test(rfscores, results))
            print 't-test Cat XGB: ', repr(stat_test(catxgbscores, results))
            print
            print
        print 'Done with: ' + clf_name
        print '====================================='
        print
        print
    ttest_df_m, ttest_df_p = pairwise_ttest(accuracy_dict)
    ttest_df_m.to_csv(os.getcwd() + '/' + str(size) +'_ttest_diff_pairs.csv')
    ttest_df_p.to_csv(os.getcwd() + '/' + str(size) +'_ttest_p_val_pairs.csv')
    res_df.to_csv(os.getcwd() + '/' + str(size) +'_results.csv')


def randomProtocolValues(X):
    # protocol features
    a = range(13) + range(66,69)
    l = len(X)

    for i in a:
        X[i] = np.random.random(l) * 100

    return X


def oneFold(X_train,y_train,X_test,y_test,clf):
    global train_times
    global test_times
    # VPN run
    # Translate labels to os/browser/os+browser
    # dataset_path = os.getcwd() + "/data_set/samples_5.11.16_nona.csv"
    # df = pd.read_csv(dataset_path)
    # y_test = df['69']
    #y_test = transform_labels(y_test, os_flag=0, browser_flag=0, app_flag=1)
    #y_train = transform_labels(y_train, os_flag=0, browser_flag=0, app_flag=1)
    ###
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    start_train = timer()
    clf.fit(X_train, y_train)
    end_train = timer()
    train_times.append(end_train - start_train)
    print 'Train time: ', end_train - start_train
    start_test = timer()
    fold_score = clf.score(X_test, y_test)
    print 'Fold score: ' + repr(fold_score)
    end_test = timer()
    test_times.append(end_test - start_test)
    print 'Test time: ', end_test - start_test
    return fold_score

def loadData(size):
    # See ShuffleSplit below
    # Load data
    data_path = os.path.join(os.getcwd(),'data_set')
    #data_fname = 'time_segments_13.12.16_nona.csv'
    #data_fname = 'primary_secondary_13.12.16_nona.csv'
    data_fname = 'mobile_actions_14.12.16_nona.csv'
    df = pd.read_csv(os.path.join(data_path,data_fname),header=None)
    df = df.reindex(np.random.permutation(df.index))
    print df 
    y = df[28]
    df = df.drop([0,28], axis=1)

    return [(df.iloc[train], y.iloc[train], df.iloc[test], y.iloc[test]) for train, test in KFold(len(df),5,shuffle=True)]

def getFeatureGroups():

    group_names = ['Hist, Peak, Size_no_flow']

    columns_to_drop = [[]]

    return group_names, columns_to_drop

def loadData_real_time(size):
	"""fiveFold expects 5 permutations of X_train y_train X_test y_test """
	data_path = os.path.join(os.getcwd(),'data_set','samples_first_10_min_16.11.16_nona.csv')
	df = pd.read_csv(data_path, index_col=False)
	y = df['69']
	df = df.drop(['Unnamed: 0','69'],axis=1)
	df.columns = [int(v) for v in df.columns]
	kf = KFold(len(df),5,shuffle=True)
	inds = []
	for train, test in kf:
		inds.append((train,test))
	train_0, test_0 = inds[0]
	train_1, test_1 = inds[1]
	train_2, test_2 = inds[2]
  	train_3, test_3 = inds[3]
	train_4, test_4 = inds[4]
	X_train_0, y_train_0, X_test_0, y_test_0 = df.iloc[train_0], y.iloc[train_0], df.iloc[test_0], y.iloc[test_0]
    	X_train_1, y_train_1, X_test_1, y_test_1 = df.iloc[train_1], y.iloc[train_1], df.iloc[test_1], y.iloc[test_1]
    	X_train_2, y_train_2, X_test_2, y_test_2 = df.iloc[train_2], y.iloc[train_2], df.iloc[test_2], y.iloc[test_2]
    	X_train_3, y_train_3, X_test_3, y_test_3 = df.iloc[train_3], y.iloc[train_3], df.iloc[test_3], y.iloc[test_3]
    	X_train_4, y_train_4, X_test_4, y_test_4 = df.iloc[train_4], y.iloc[train_4], df.iloc[test_4], y.iloc[test_4]

	return [(X_train_0, y_train_0, X_test_0, y_test_0),
            	(X_train_1, y_train_1, X_test_1, y_test_1),
            	(X_train_2, y_train_2, X_test_2, y_test_2),
           	(X_train_3, y_train_3, X_test_3, y_test_3),
            	(X_train_4, y_train_4, X_test_4, y_test_4)]


def loadData_org(size):
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

    if(size > 0):

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

    else:
	
	df_train_0 = pd.DataFrame(X_train_0.toarray())
	df_test_0 = pd.DataFrame(X_test_0.toarray())

	df_train_1 = pd.DataFrame(X_train_1.toarray())
	df_test_1 = pd.DataFrame(X_test_1.toarray())

	df_train_2 = pd.DataFrame(X_train_2.toarray())
	df_test_2 = pd.DataFrame(X_test_2.toarray())

	df_train_3 = pd.DataFrame(X_train_3.toarray())
	df_test_3 = pd.DataFrame(X_test_3.toarray())


	df_train_4 = pd.DataFrame(X_train_4.toarray())
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

def getFeatureGroups_org():
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
    # VPN run
    # Inject VPN dataset as test-set
    # dataset_path = os.getcwd() + "/data_set/samples_5.11.16_nona.csv"
    # df = pd.read_csv(dataset_path, header=None, skiprows=1)
    # X_test = df.drop(69,axis=1)
    ###
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

#-----------------------MAIN------------------------------------------------------
if __name__ == "__main__":

    #sizes = [50, 100, 200, 300, 400, 500,
    #         600, 700, 800, 900, 1000,2000,5000,10000,-1]
    sizes = [-1]
    for size in sizes:
        fiveFold(size)
