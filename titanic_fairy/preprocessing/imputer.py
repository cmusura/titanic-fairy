"""Modulo para imputar valores faltantes en los datos"""

from titanic_fairy.enums.titanic_fields import Fields, Preprocess_
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class Imputer(BaseEstimator, TransformerMixin):
    """Imputacion de datos faltantes en base a distintas estrategias (moda y promedio)

    Las estrategias y los features considerados en cada caso se encuentran en el modulo Enum.

    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Sacamos las instrucciones del modulo enum
        # Cada feature considerado tiene una estrategia de imputacion
        instruct = Preprocess_.Impute.value
        for feature in instruct.keys():
            imputer = SimpleImputer(strategy=instruct[feature])
            X[feature] = imputer.fit_transform(X[[feature]]).flatten()
        return X
