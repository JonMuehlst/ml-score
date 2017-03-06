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

def read_data_set(path, rows_to_skip):
    ds = pd.read_csv(path,sep='\t', skiprows=rows_to_skip, header=None)
    return ds

def trainKNN(X_train,y_train):
    #normalization
    scaler = preprocessing.StandardScaler().fit(X_train)
    filename = 'KNNTrainScalar.joblib.pkl'
    joblib.dump(scaler, filename, compress=9)
    XTrainScaled = scaler.transform(X_train)

    # cross fitting
    # neighbors_range = [2,4,8,16,32,64]
    neighbors_range = [16,32]

    distance_types = ['chebyshev', 'sokalmichener',
    'canberra', #'haversine',
    #'rogerstanimoto', 'matching',
    'dice', 'euclidean',
    'braycurtis', 'russellrao',
    'cityblock', 'manhattan',
    #'infinity', 'jaccard',
    #'sokalsneath', # 'seuclidean',
    #'kulsinski', 'minkowski',
    #'mahalanobis', 'p',
    #'l2', 'hamming',
    #'l1', #'wminkowski',
    #'pyfunc']
    ]



    algorithms=['ball_tree']

    param_grid = dict(algorithm=algorithms, n_neighbors=neighbors_range, metric=distance_types)
    cv = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,
                        n_jobs=-1, cv=cv, verbose=100)
    grid.fit(XTrainScaled, y_train)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    # train by best params
    n_neighbors = grid.best_params_['n_neighbors']
    metric = grid.best_params_['metric']

    clf_opt = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    clf_opt.fit(XTrainScaled, y_train)

    filename = 'KNN.joblib.pkl'
    joblib.dump(clf_opt, filename, compress=9)

def testKNN(X_test,y_test):
    clf = joblib.load('KNN.joblib.pkl')
    scaler = joblib.load('KNNTrainScalar.joblib.pkl')
    X_testScaled = scaler.transform(X_test)
    y_pred = clf.predict(X_testScaled)
    print('KNN precision: ',metrics.precision_score(y_test, y_pred))
    print(' KNN accuracy: ',metrics.accuracy_score(y_test, y_pred))
    precision, recall, threshold = precision_recall_curve(y_test, y_pred)

def trainSVC_RBF(X_train,y_train):
    #normalization
    scaler = preprocessing.StandardScaler().fit(X_train)
    filename = 'TrainScalar.joblib.pkl'
    joblib.dump(scaler, filename, compress=9)
    XTrainScaled = scaler.transform(X_train)

    # cross fitting
    C_range = [2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(5),2**(5),2**(7),2**(9),2**(11),2**(13),2**(15)]
    gamma_range = [2**(-15),2**(-13),2**(-11),2**(-9),2**(-7),2**(-5),2**(-3),2**(-1),2**(1),2**(3),2**(3)]
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(XTrainScaled, y_train)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    C=grid.best_params_['C']
    clf_ChromeTrainRBF = rbf_svc = SVC(kernel='rbf', gamma=grid.best_params_['gamma'], C=C)
    clf_ChromeTrainRBF.fit(XTrainScaled, y_train)
    #MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
    filename = 'SVMRBF.joblib.pkl'
    joblib.dump(clf_ChromeTrainRBF, filename, compress=9)

def testSVC_RBF(X_test,y_test):
    clf = joblib.load('SVMRBF.joblib.pkl')
    scaler = joblib.load('TrainScalar.joblib.pkl')
    X_testScaled = scaler.transform(X_test)
    y_pred = clf.predict(X_testScaled)
    print('SVC percision: ',metrics.precision_score(y_test, y_pred))
    print('svc accuracy: ',metrics.accuracy_score(y_test, y_pred))
    precision, recall, threshold = precision_recall_curve(y_test, y_pred)

def plot_confusion_matrix(cm, eunique_label,title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(eunique_label))
    plt.xticks(tick_marks, eunique_label, rotation=45)
    plt.yticks(tick_marks, eunique_label)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def trainTestM(X_train,X_test,y_train,y_test):
    h = .02  # step size in the mesh

    names = [ "Nearest Neighbors", "Linear SVM", "RBF SVM","Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
    classifiers = [
    KNeighborsClassifier(32),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    # DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        print('name: ',name,'score: ',score)
        print(name,' percision: ',metrics.precision_score(y_test, y_pred))
        print(name,' accuracy: ',metrics.accuracy_score(y_test, y_pred))
        class_report  = classification_report(y_test, y_pred)
        out_name= 'accuracy_report_{}_.txt'.format(name)
        with open(out_name, "w") as text_file:
            text_file.write(class_report)
            text_file.write('percision: %s'%metrics.precision_score(y_test, y_pred))
            text_file.write(' accuracy: %s'%metrics.accuracy_score(y_test, y_pred))
        print(class_report)

        filename = "clas_{}".format(name)
        joblib.dump(clf, filename, compress=9)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(cm_normalized)
        plt.figure()
        eunique_label = np.unique(y_train.tolist())
        plot_confusion_matrix(cm_normalized, eunique_label,title='Normalized confusion matrix')
        image_name = name + ".pdf"
        plt.savefig(image_name)

def ensembleVoting(X_train,y_train,X_test,y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # num_folds = 3
    # num_instances = len(X)
    # seed = 7
    # kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # create the sub models
    estimators = []
    # model1 = LogisticRegression()
    # estimators.append(('logistic', model1))
    # model2 = DecisionTreeClassifier()
    # estimators.append(('cart', model2))
    # model3 = SVC()
    # estimators.append(('svm', model2))
    # names = [ "Nearest Neighbors", "Linear SVM", "RBF SVM","Decision Tree",
    #      "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
    #      "Quadratic Discriminant Analysis"]

    model1 = KNeighborsClassifier()
    estimators.append(('knn', model1))
    model2 = SVC()
    estimators.append(('svmrbf', model2))
    model3 = DecisionTreeClassifier(max_depth=20)
    estimators.append(('DecisionTree', model3))
    # model4 = LinearDiscriminantAnalysis()
    # estimators.append(('LDA', model4))

    # classifiers = [
    # KNeighborsClassifier(32),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # LinearDiscriminantAnalysis(),
    # QuadraticDiscriminantAnalysis()]

    # create the ensemble model
    ensemble = VotingClassifier(estimators)#, voting='soft', weights=[1,2,1,1])
    # ensemble.fit(X_train,y_train)
    params = {'svmrbf__gamma': [2**(-7),2**(-5),2**(-3)],
              'svmrbf__C': [2**(5),2**(7),2**(9),2**(11),2**(13)],
              'knn__algorithm': ['ball_tree'],
              'knn__n_neighbors': [14, 16, 20],
              'knn__metric': [ # 'chebyshev', 'sokalmichener',
              'canberra'#, 'dice', 'euclidean',
              #'braycurtis', 'russellrao','cityblock', 'manhattan']}
              ]}

    grid = GridSearchCV(estimator=ensemble, param_grid=params, cv=5, n_jobs=-1)
    grid = grid.fit(X_train,y_train)
    print(grid.grid_scores_)
    # print(ensemble.score(X_test,y_test))
    # results = cross_validation.cross_val_score(ensemble, X, y, cv=kfold)
    # print(results.mean())

def CategoricalEnsembleVoting(X_train,y_train,X_test,y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = CategoryClassifier()
    clf.fit(X_train, y_train)
    print(repr(clf.score(X_test, y_test)))

""" mean: 0.99819, std: 0.00051, params:
    {'svmrbf__gamma': 0.0078125, 'knn__algorithm': 'ball_tree',
    'knn__n_neighbors': 16, 'svmrbf__C': 8192, 'knn__metric': 'canberra'} """
def fiveFold():

    # Feature groups
    # protocol_dependent = range(13) + range(66,69)
    # protocol_dependent = range(23) + range(66,69)
    # peak features
    # protocol_dependent = range(23,41)
    # All but peak
    # protocol_dependent = range(23) + range(41,69)
    fsslv_cipher_suites = [6,7,8,9,10,11,12]
    protocol_dependent = []

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

    X_train_0 = df_train_0.drop(protocol_dependent, axis=1)
    X_test_0 = df_test_0.drop(protocol_dependent, axis=1)
    X_train_1 = df_train_1.drop(protocol_dependent, axis=1)
    X_test_1 = df_test_1.drop(protocol_dependent, axis=1)
    X_train_2 = df_train_2.drop(protocol_dependent, axis=1)
    X_test_2 = df_test_2.drop(protocol_dependent, axis=1)
    X_train_3 = df_train_3.drop(protocol_dependent, axis=1)
    X_test_3 = df_test_3.drop(protocol_dependent, axis=1)
    X_train_4 = df_train_4.drop(protocol_dependent, axis=1)
    X_test_4 = df_test_4.drop(protocol_dependent, axis=1)

    # X_train_0 = randomProtocolValues(X_train_0)
    # X_test_0 = randomProtocolValues(X_test_0)
    # X_train_1 = randomProtocolValues(X_train_1)
    # X_test_1 = randomProtocolValues(X_test_1)
    # X_train_2 = randomProtocolValues(X_train_2)
    # X_test_2 = randomProtocolValues(X_test_2)
    # X_train_3 = randomProtocolValues(X_train_3)
    # X_test_3 = randomProtocolValues(X_test_3)
    # X_train_4 = randomProtocolValues(X_train_4)
    # X_test_4 = randomProtocolValues(X_test_4)


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

    # ensemble = VotingClassifier(estimators,voting='hard')
    ensemble = CategoryClassifier()

    # CategoricalEnsembleVoting(X_train_0, y_train_0, X_test_0, y_test_0)
    oneFold(X_train_0, y_train_0, X_test_0, y_test_0, ensemble)
    oneFold(X_train_1, y_train_1, X_test_1, y_test_1, ensemble)
    oneFold(X_train_2, y_train_2, X_test_2, y_test_2, ensemble)
    oneFold(X_train_3, y_train_3, X_test_3, y_test_3, ensemble)
    oneFold(X_train_4, y_train_4, X_test_4, y_test_4, ensemble)


def randomProtocolValues(X):
    # protocol features
    a = range(13) + range(66,69)
    l = len(X)

    for i in a:
        X[i] = np.random.random(l) * 100

    return X



def oneFold(X_train,y_train,X_test,y_test,clf):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)
    print(repr(clf.score(X_test, y_test)))

#-----------------------MAIN------------------------------------------------------
if __name__ == "__main__":

    data_path = os.getcwd() + "/data_set"
    all_features_path = data_path + "/samples_25.2.16_all_features_triple.csv"
    # all_features_path = data_path + "/samples_17.7.16_all_features_app.csv"
    rows_to_skip = [0]
    ds = read_data_set(all_features_path, rows_to_skip=rows_to_skip)
    # print repr(ds)

    ds = ds.dropna()
    y = ds.iloc[:,len(ds.columns)-1]
    # num_classes, y = np.unique(y, return_inverse=True)
    # X = ds.drop([str(len(ds.columns)-1)], axis=1)
    X = ds.drop([len(ds.columns)-1], axis=1)
    # X = ds.drop([2,4,9,11], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # trainTestM(X_train,X_test,y_train,y_test)
    # trainKNN(X_train,y_train)
    # testKNN(X_test,y_test)
    # ensembleVoting(X_train, y_train, X_test, y_test)
    # CategoricalEnsembleVoting(X_train, y_train, X_test, y_test)
    fiveFold()
