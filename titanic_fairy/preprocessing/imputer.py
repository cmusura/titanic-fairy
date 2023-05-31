"""Modulo para imputar valores faltantes en los datos"""

from titanic_fairy.enums.titanic_fields import Fields
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy='mean')
        X[Fields.Age] = imputer.fit_transform(X[[Fields.Age]])
        return X
    
class CabinImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy='most_frequent')
        X[Fields.Cabin] = imputer.fit_transform(X[[Fields.Cabin]])
        return X
    
class EmbarkedImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy='most_frequent')
        X[Fields.Embarked] = imputer.fit_transform(X[[Fields.Embarked]])
        return X