"""Wrapper de todos los otros modulos de preprocesamiento y los realiza en conjunto."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from titanic_fairy.preprocessing.dropper import FeatureDropper
from titanic_fairy.preprocessing.encoder import FeatureEncoder
from titanic_fairy.preprocessing.imputer import Imputer


class Preprocess(BaseEstimator, TransformerMixin):
    """Wrapper que realiza los pasos del preprocesamiento en el orden adecuado"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        preprocesseser = Pipeline(
            [
                ("imputer", Imputer()),
                ("featureencoder", FeatureEncoder()),
                ("featuredropper", FeatureDropper()),
            ]
        )
        X = preprocesseser.fit_transform(X)
        return X
