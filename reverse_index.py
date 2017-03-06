# Step 0: load data to pandas


# Step 1: hash original samples


# Step 2: save permutation indexes

import numpy as np

def loadData():
    pass

def generate_hash(df):
    return np.array(df.apply(lambda x: hash(tuple(x)), axis = 1))

def get_permutation_indexes(df, hashes):
    permu_hashes = np.array(df.apply(lambda x: hash(tuple(x)), axis = 1))
    permu_i = np.zeros(len(df))

    for i in range(len(permu_hashes)):
        permu_i[i] = hashes.index(permu_hashes[i])

    return permu_i

def reverse_index():
    loadData()
    hashes = generate_hash(df)
    permu_0 = get_permutation_indexes(df_0, hashes)
    permu_1 = get_permutation_indexes(df_0, hashes)
    permu_2 = get_permutation_indexes(df_0, hashes)
    permu_3 = get_permutation_indexes(df_0, hashes)
    permu_4 = get_permutation_indexes(df_0, hashes)
