import numpy as np
import tqdm
import pickle
from copy import deepcopy
import pandas as pd


def read_data(data_id_list):
    all_data_list = []
    all_label_list = []
    count_set_x = 0
    count_set = [0]
    for i in tqdm.tqdm(data_id_list):
        try:
            # with open(f'feature_engineering/data/{features_file}/bulk_features_{i}.pkl', 'rb') as f:
            #     bulk_features = pickle.load(f)
            with open(f'feature_engineering/data/{features_file}/sc_features_{i}.pkl', 'rb') as f:
                sc_features = pickle.load(f)
            with open(f'feature_engineering/data/samples/label_{i}.pkl', 'rb') as f:
                label_list = pickle.load(f)
            # bulk_features.columns = bulk_features.columns + '(bulk)'
            # all_features = pd.concat([bulk_features, sc_features], axis=1)
            all_features = sc_features

            tmp = 0. * deepcopy(all_features)
            for col_name, col in all_features.iteritems():
                if "A->B" in col_name:
                    tmp[col_name.replace("A->B", "B->A")] = col
                elif "B->A" in col_name:
                    tmp[col_name.replace("B->A", "A->B")] = col
                elif "difference" in col_name:
                    tmp[col_name] = -col
                else:
                    raise Exception("columns name must be in A->B, B->A or difference!")

            # big_df = all_features
            big_df = pd.concat((all_features, tmp))
            all_data_list.append(big_df)

            label_list2 = []
            for x_gene, y_gene, label in label_list:
                if label == 0:
                    new_label = 3
                elif label == 1:
                    new_label = 2
                else:
                    raise ValueError('label must be 0 or 1!')
                label_list2.append([y_gene, x_gene, new_label])
            label_list = label_list + label_list2
            count_set_x += len(label_list)
            count_set.append(count_set_x)
            label_data = pd.DataFrame(label_list, columns=['x_gene', 'y_gene', 'label'])
            all_label_list.append(label_data)
        except FileNotFoundError as e:
            print(e)
    all_data = pd.concat(all_data_list)
    all_label = pd.concat(all_label_list)
    all_data.index = range(all_data.shape[0])
    all_label.index = range(all_label.shape[0])
    return all_data, all_label, count_set


features_file = 'features'
TEST_NUM = 851
# TEST_NUM = 3057

whole_data = [i for i in range(TEST_NUM)]
test_indel = 2
test_list = [i for i in
             range(int(np.ceil((test_indel - 1) * 0.333 * TEST_NUM)), int(np.ceil(test_indel * 0.333 * TEST_NUM)))]
train_list = [i for i in whole_data if i not in test_list]

x_train, y_train, _ = read_data(train_list)
x_test, y_test, count_set = read_data(test_list)

x_test, y_test, count_set = read_data(whole_data)
xxxxx = x_test.copy()
yyyyy = y_test.copy()
x_train = xxxxx.iloc[5000:]
y_train = yyyyy.iloc[5000:]
x_test = xxxxx.iloc[:5000]
y_test = yyyyy.iloc[:5000]

with open('./feature_engineering/data/x_train.pkl', 'wb') as f:
    pickle.dump(x_train, f)
with open('./feature_engineering/data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('./feature_engineering/data/x_test.pkl', 'wb') as f:
    pickle.dump(x_test, f)
with open('./feature_engineering/data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

x_train.head(100).to_csv('./feature_engineering/data/x_train(sample).csv', float_format='%.5g')
y_train.head(100).to_csv('./feature_engineering/data/y_train(sample).csv', float_format='%.5g')
