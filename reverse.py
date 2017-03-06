def generate_hash(df):
    return np.array(df.apply(lambda x: hash(tuple(x)), axis = 1))

def get_permutation_indexes(df, hashes):
    permu_hashes = np.array(df.apply(lambda x: hash(tuple(x)), axis = 1))
    permu_i = np.zeros(len(df))

    for i in range(len(permu_hashes)):
        permu_i[i] = hashes.index(permu_hashes[i])

    return permu_i
