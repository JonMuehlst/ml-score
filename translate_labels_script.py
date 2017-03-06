#!/usr/bin/env python
from translate_labels import transform_to_os_labels, transform_to_browser_labels, transform_to_app_labels
import pandas as pd
import os
import copy

if __name__ == "__main__":
    data_set_path = os.getcwd() + '/data_set'
    path = data_set_path + '/samples_25.2.16_all_features_triple.csv'
    ds = pd.read_csv(path,sep='\t',header=None,skiprows=[0])
    labels_column = len(ds.columns)-1
    ds[labels_column] = ds[labels_column].astype(int)
    ds_app = ds_browser = ds_os = ds
    y_triple = copy.deepcopy(ds[labels_column])

    ds[labels_column] = transform_to_os_labels(y_triple)
    ds.to_csv(data_set_path + '/samples_17.7.16_all_features_os.csv', sep='\t', index=False)

    ds[labels_column] = transform_to_browser_labels(y_triple)
    ds.to_csv(data_set_path + '/samples_17.7.16_all_features_browser.csv', sep='\t', index=False)

    ds[labels_column] = transform_to_app_labels(y_triple)
    ds.to_csv(data_set_path + '/samples_17.7.16_all_features_app.csv', sep='\t', index=False)
