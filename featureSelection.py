print(__doc__)

from time import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier
import os

from featureClassification import read_data_set
# Number of cores to use to perform parallel fitting of the forest model
n_jobs = -1


data_path = os.getcwd() + "/data_set"
all_features_path = data_path + "/samples_25.2.16_all_features_triple.csv"

rows_to_skip = [0]
ds = read_data_set(all_features_path, rows_to_skip=rows_to_skip)

ds = ds.dropna()
y = ds.iloc[:,len(ds.columns)-1]
X = ds.drop([len(ds.columns)-1], axis=1)

# Load the faces dataset
# data = fetch_olivetti_faces()
# X = data.images.reshape((len(data.images), -1))
# y = data.target

# mask = y < 5  # Limit to 5 classes
# X = X[mask]
# y = y[mask]

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# Build a forest and compute the pixel importances
print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
t0 = time()
forest = ExtraTreesClassifier(n_estimators=1000,
                              max_features=69,
                              n_jobs=n_jobs,
                              random_state=0)

forest.fit(X, y)
print("done in %0.3fs" % (time() - t0))
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
