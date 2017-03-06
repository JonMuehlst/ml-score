from translate_labels import transform_to_os_labels, transform_to_browser_labels, transform_to_app_labels

import copy
from sklearn.ensemble import VotingClassifier
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

from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class CategoryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        # print(y)
        y_os = transform_to_os_labels(copy.deepcopy(y))
        y_browser = transform_to_browser_labels(copy.deepcopy(y))
        y_app = transform_to_app_labels(copy.deepcopy(y))

        # print(y_os)
        # print(y_browser)
        # print(y_app)

        self.os_ensemble = VotingClassifier(self.getEstimators())
        self.os_ensemble.fit(X,y_os)

        self.browser_ensemble = VotingClassifier(self.getEstimators())
        self.browser_ensemble.fit(X,y_browser)

        self.app_ensemble = VotingClassifier(self.getEstimators())
        self.app_ensemble.fit(X,y_app)

        return self

    def predict(self, X):
        y_pred_os = self.os_ensemble.predict(X)
        y_pred_browser = self.browser_ensemble.predict(X)
        y_pred_app = self.app_ensemble.predict(X)
        ans = [ int(10000+ai+bi+ci) for ai,bi,ci in zip(y_pred_os,
                                                    y_pred_browser,y_pred_app)]
        # print repr(ans)
        return ans
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def getEstimators(self):
        estimators = []
        # model1 = KNeighborsClassifier(n_neighbors=16,algorithm='ball_tree',
        #                                 metric='canberra', n_jobs=-1)
        # estimators.append(('knn', model1))
        # model2 = SVC(gamma=0.0078125,C=8192, probability=False)
        # estimators.append(('svmrbf', model2))
        # model3 = DecisionTreeClassifier()#max_depth=50)
        # estimators.append(('DecisionTree', model3))
        # model4 = RandomForestClassifier(n_estimators=1000, oob_score=True,
        #                                 n_jobs=-1)
        # estimators.append(('RandomForest', model4))
        model5 = XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.1)
        estimators.append(('XGBoost', model5))
        return estimators

    def get_params(self, deep=False):
        model5 = XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.1)
        return model5.get_params(deep=deep)
