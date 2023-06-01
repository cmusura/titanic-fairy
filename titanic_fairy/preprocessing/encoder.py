"""Modulo encargado de imputar. """
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from titanic_fairy.enums.titanic_fields import Fields, Preprocess_


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Toma nuesta tabla y codifica las columnas categoricas para ser procesadas por el modelo

    Las estrategias y los features considerados en cada caso se encuentran en el modulo Enum.

    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ocupamos OneShotEncoding
        encoder = OneHotEncoder()

        # Recuperamos las variables a codificar en Enum
        features_to_encode = Preprocess_.Encode.value

        for feature in features_to_encode:
            # Codificamos los valores
            matrix = encoder.fit_transform(X[[feature]]).toarray()

            # Muy importante ordenar los valores en esta linea (Si no la asignacion puede no ser la correcta)
            new_column_names = sorted(X[feature].unique())

            # Generamos nuevas columnas One-Shot
            for i in range(len(matrix.T)):
                X[new_column_names[i]] = matrix.T[i]

        return X
