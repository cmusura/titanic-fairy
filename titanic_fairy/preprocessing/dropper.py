"""Modulo para desechar columnas que no seran usadas en le modelo."""
from sklearn.base import BaseEstimator, TransformerMixin
from titanic_fairy.enums.titanic_fields import Fields, Preprocess_


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Desecha las variables que son consideradas inutiles para el modelo"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        to_drop = Preprocess_.Drop.value
        return X.drop(to_drop, axis=1)
