import pandas as pd
import pickle
from feature_engineering.random_functions import preprocess

import os

missing_data = []
for i in range(3000):
    if not os.path.exists(f'feature_engineering/data/features2/sc_features_{i}.pkl'):
        missing_data.append(i)

TEST_NUM = 3057
NUM = 3057
file_error = []
# for i in missing_data:
for i in range(933, NUM):
    try:
        with open(f'feature_engineering/data/samples/bulk_{i}.pkl', 'rb') as f:
            bulk_pairs = pickle.load(f)
        with open(f'feature_engineering/data/samples/sc_{i}.pkl', 'rb') as f:
            sc_pairs = pickle.load(f)

        # bulk_store = pd.DataFrame(bulk_pairs)
        # del bulk_store[0]
        # del bulk_store[1]
        # # store = bulk_store
        # bulk_features = preprocess(bulk_store)

        sc_store = pd.DataFrame(sc_pairs)
        del sc_store[0]
        del sc_store[1]
        sc_features = preprocess(sc_store)

        # with open(f'feature_engineering/data/features2/bulk_features_{i}.pkl', 'wb') as f:
        #     pickle.dump(bulk_features, f)

        with open(f'feature_engineering/data/features/sc_features_{i}.pkl', 'wb') as f:
            pickle.dump(sc_features, f)
    except FileNotFoundError:
        file_error.append(i)
