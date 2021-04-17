import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA

from feature_engineering.utils import is_categorical
from feature_engineering.gap_statistic import FittedMiniBatchKMeans


class NumericalToCategorical(object):

    def __init__(self, clustering=None, min_clusters=2, verify=True):
        """Takes in a clustering classifier in order to convert numerical features into categorical.
        """
        if clustering is None:
            clustering = FittedMiniBatchKMeans(min_clusters)
        self.clustering = clustering
        self.verify = verify

    def fit(self, x):
        reshaped = x.reshape(-1, 1)
        self.clustering.fit(reshaped)

    def transform(self, x):
        reshaped = x.reshape(-1, 1)
        result = self.clustering.predict(reshaped)
        assert result.shape == x.shape
        return result

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class CategoricalToNumerical(object):

    def __init__(self, dimensionality_reducer=None, verify=True):
        pass
        """Takes in a dimensionality reducer in order to convert categorical features into numerical.
        """
        if dimensionality_reducer is None:
            dimensionality_reducer = PCA(1, svd_solver='randomized')
        self.dimensionality_reducer = dimensionality_reducer
        self.verify = verify
        self.binarizer = LabelBinarizer()

    def fit(self, x):
        self._verify(x, self.verify)
        binarized = self.binarizer.fit_transform(x)
        self.dimensionality_reducer.fit(binarized)

    def transform(self, x):
        self._verify(x, False)
        binarized = self.binarizer.transform(x)
        result = self.dimensionality_reducer.transform(binarized).flatten()
        assert x.shape == result.shape
        return result

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    @staticmethod
    def _verify(x, verify):
        if verify:
            assert is_categorical(x)
        else:
            assert isinstance(x, np.ndarray)
            assert len(x.shape) == 1
