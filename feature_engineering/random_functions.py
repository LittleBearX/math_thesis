from multiprocessing import Pool
from functools import partial
import pandas as pd
import numpy as np

from feature_engineering.convert import NumericalToCategorical
from feature_engineering.binary_features import BINARY_FEATURES
from feature_engineering.unary_features import UNARY_FEATURES
from feature_engineering.classification_machines import CLASSIFICATION_FEATURES
from feature_engineering.regression_machines import REGRESSION_FEATURES
from feature_engineering.utils import print_current_time

PARALLEL = True
# PARALLEL = False


def feature_map(func_name, func, list_of_features):
    new_features = []
    len0 = len(list_of_features[0])
    for features in list_of_features:
        assert len(features) == len0
    for f in zip(*list_of_features):
        names, values = zip(*f)
        feature_name = names[0]
        for n in names:
            assert n == feature_name
        new_name = "{}_{}".format(feature_name, func_name)
        new_features.append((new_name, func(values)))
    return new_features


def feature_difference(f1, f2):
    def diff(values):
        assert len(values) == 2
        return values[0] - values[1]

    return feature_map("difference", diff, [f1, f2])


def feature_sum(list_of_features):
    return feature_map("sum", sum, list_of_features)


def feature_avg(list_of_features):
    return feature_map("average", np.mean, list_of_features)


def combine_features(list_of_features):
    names = []
    values = []
    for features in list_of_features:
        tmp_names, tmp_values = zip(*features)
        names += tmp_names
        values += tmp_values
    return names, values


def preprend_name(pre, features):
    return [("{}_{}".format(pre, name), val) for name, val in features]


def convert_to_categorical(data):
    assert isinstance(data, np.ndarray)
    NUM_CATEGORIES = 10
    rows = data.shape[0]
    new_data = np.zeros(rows)
    percentile = step = 100.0 / NUM_CATEGORIES
    while percentile < 100.0:
        new_data += data > np.percentile(data, percentile)
        percentile += step
    return new_data


def convert_to_numerical(data):
    assert isinstance(data, np.ndarray)
    # return [ss.fit_transform(data)]
    return (data - data.mean()) / (data.std() + 1e-8)


def preprocess(store):
    FEATURES = BINARY_FEATURES + UNARY_FEATURES + REGRESSION_FEATURES + CLASSIFICATION_FEATURES
    # FEATURES = BINARY_FEATURES + UNARY_FEATURES

    pool = Pool(5)
    # V2_cache = create_V2_cache(store, pool)

    features = []
    for count, feature in enumerate(FEATURES):
        name, func, func_args = feature[0], feature[1], feature[2:]
        print(count, end='\t')
        print(name, end=' ')
        print_current_time()
        tmp_feature = feature_creation_V1(pool, store, func, func_args, name)
        features.append(tmp_feature)
        # tmp_feature2 = feature_creation_V2(pool, V2_cache, func, func_args, name)
        # tmp_feature2.columns = 'V2_' + tmp_feature2.columns
        # features.append(tmp_feature2)
    pool.close()
    return pd.concat(features, axis=1)


def create_V2_cache(store, pool):
    if PARALLEL:
        V2_cache = pool.map(create_V2_cache_transform, store.values)
    else:
        V2_cache = list(map(create_V2_cache_transform, store.values))
    return V2_cache


def create_V2_cache_transform(row):
    a, b = row
    num_x, cat_x, num_y, cat_y = a, a, b, b

    if num_x.std() > 1e-5:
        cat_x = NumericalToCategorical(verify=False).fit_transform(num_x)
    else:
        cat_x = np.zeros_like(num_x, dtype='int')
    if num_y.std() > 1e-5:
        cat_y = NumericalToCategorical(verify=False).fit_transform(num_y)
    else:
        cat_y = np.zeros_like(num_y, dtype='int')

    return num_x, cat_x, num_y, cat_y


def feature_creation_V1(pool, store, func, func_args, name):
    desired_type = name[:2]
    assert desired_type in ["NN", "NC", "CN", "CC"]
    new_func = partial(feature_creation_row_helper, func, func_args, desired_type)
    if PARALLEL:
        mapped = pool.map(new_func, store.values)
    else:
        mapped = [new_func(i) for i in store.values]

    names = None
    transformed = []
    for row_names, transformed_row in mapped:
        if names is None:
            names = row_names
        assert names == row_names
        transformed.append(transformed_row)
    new_names = ["{}_{}".format(name, n) for n in names]
    result = pd.DataFrame(transformed, columns=new_names).fillna(0)
    result[np.isinf(result)] = 0
    return result


def feature_creation_row_helper(func, func_args, desired_type, row):
    # row = store.values[0]
    if len(func_args) > 0:
        func = func(*func_args)
    return feature_creation_row(func, desired_type, row)


def feature_creation_row(func, desired_type, row):
    x, y = row
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    cat_x, cat_y = convert_to_categorical(x), convert_to_categorical(y)
    num_x, num_y = convert_to_numerical(x), convert_to_numerical(y)
    left = asymmetric_feature_creation(func, desired_type, num_x, cat_x, num_y, cat_y)
    # right = asymmetric_feature_creation(func, desired_type, num_x, cat_x, num_y, cat_y)
    right = asymmetric_feature_creation(func, desired_type, num_y, cat_y, num_x, cat_x)
    relative = feature_difference(left, right)
    new_left = preprend_name("A->B", left)
    new_right = preprend_name("B->A", right)
    features = (new_left, new_right, relative)
    return combine_features(features)


def asymmetric_feature_creation(func, desired_type, num_x, cat_x, num_y, cat_y):
    if desired_type == "NN":
        features = func(num_x, num_y)
    elif desired_type == "CN":
        features = func(cat_x, num_y)
    elif desired_type == "NC":
        features = func(num_x, cat_y)
    elif desired_type == "CC":
        features = func(cat_x, cat_y)
    else:
        raise Exception("Incorrect desired type: {}".format(desired_type))
    return features


def feature_creation_V2(pool, V2_cache, func, func_args, name):
    desired_type = name[:2]
    assert desired_type in ["NN", "NC", "CN", "CC"]

    new_func = partial(feature_creation_row_helper_V2, func, func_args, desired_type)
    if PARALLEL:
        mapped = pool.map(new_func, V2_cache)
    else:
        mapped = [new_func(i) for i in V2_cache]

    names = None
    transformed = []
    for row_names, transformed_row in mapped:
        if names is None:
            names = row_names
        assert names == row_names
        transformed.append(transformed_row)
    new_names = ["{}_{}".format(name, n) for n in names]
    result = pd.DataFrame(transformed, columns=new_names).fillna(0)
    result[np.isinf(result)] = 0
    return result


def feature_creation_row_helper_V2(func, func_args, desired_type, row):
    if len(func_args) > 0:
        func = func(*func_args)
    row = [x.astype(np.float) for x in row]
    return feature_creation_row_V2(func, desired_type, row)


def feature_creation_row_V2(func, desired_type, row):
    num_x, cat_x, num_y, cat_y = row
    assert isinstance(num_x, np.ndarray)
    assert isinstance(cat_x, np.ndarray)
    assert isinstance(num_y, np.ndarray)
    assert isinstance(cat_y, np.ndarray)

    left = asymmetric_feature_creation_V2(func, desired_type, num_x, cat_x, num_y, cat_y)
    right = asymmetric_feature_creation_V2(func, desired_type, num_y, cat_y, num_x, cat_x)
    relative = feature_difference(left, right)
    new_left = preprend_name("A->B", left)
    new_right = preprend_name("B->A", right)
    features = (new_left, new_right, relative)
    return combine_features(features)


def asymmetric_feature_creation_V2(func, desired_type, num_x, cat_x, num_y, cat_y):
    if desired_type == "NN":
        features = func(num_x, num_y)
    elif desired_type == "CN":
        features = func(cat_x, num_y)
    elif desired_type == "NC":
        features = func(num_x, cat_y)
    elif desired_type == "CC":
        features = func(cat_x, cat_y)
    else:
        raise Exception("Incorrect desired type: {}".format(desired_type))
    return features


def metafeature_creation(df):
    def or_(t1, t2):
        return ((t1 + t2) > 0) + 0.0

    def and_(t1, t2):
        return ((t1 + t2) == 2) + 0.0

    types = ["Binary", "Numerical", "Categorical"]
    assert isinstance(df, pd.DataFrame)
    a_type = np.array(df['A type'])
    b_type = np.array(df['B type'])
    metafeatures = []
    columns = []

    for t in types:
        tmp = (a_type == t) + 0.0
        columns.append("aIs" + t)
        metafeatures.append(tmp)

    for t in types:
        tmp = (a_type != t) + 0.0
        columns.append("aIsNot" + t)
        metafeatures.append(tmp)

    for t in types:
        tmp = (b_type == t) + 0.0
        columns.append("bIs" + t)
        metafeatures.append(tmp)

    for t in types:
        tmp = (b_type != t) + 0.0
        columns.append("bIsNot" + t)
        metafeatures.append(tmp)

    for t1 in types:
        for t2 in types:
            tmp = and_(a_type == t1, b_type == t2)
            columns.append("abAre" + t1 + t2)
            metafeatures.append(tmp)
            if t1 <= t2:
                tmp = or_(and_(a_type == t1, b_type == t2), and_(a_type == t2, b_type == t1))
                columns.append("abAreAmong" + t1 + t2)
                metafeatures.append(tmp)

    six_options = or_(a_type == "Binary", b_type == "Binary") + 2 * or_(a_type == "Categorical",
                                                                        b_type == "Categorical") + 3 * and_(
        a_type == "Binary", b_type == "Binary") + 3 * and_(a_type == "Categorical", b_type == "Categorical")

    columns.append("allTypes")
    metafeatures.append(six_options)

    return metafeatures, columns


def add_metafeatures(df, df_feat):
    metafeatures, columns = metafeature_creation(df)
    assert len(metafeatures) == len(columns)
    for mf, col in zip(metafeatures, columns):
        df_feat["metafeature_" + col] = mf
