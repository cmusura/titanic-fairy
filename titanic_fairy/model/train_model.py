"""Modulo para entrenar y exportar un modelo de clasificacion para el problema de Titanic. """
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from titanic_fairy.enums.titanic_fields import Fields, Preprocess_, Model_Fields

# Importamos la grilla de hiperparametors a revisar
param_grid_ = Model_Fields.Param_Grid.value


def build_model(X: pd.DataFrame, y: pd.Series):
    """Arma un Pipeline que dara como output el modelo de clasificacion

    El Pipeline consiste en el escalamiento de las variables seguido de 
    ajustar un RandomForest con sus hiperparametros ajustados segun un GridSearch

    :param X: DataFrame con los datos de entrenamiento 
    :type X: pd.DataFrame
    :param y: Serie de pandas con el valor target
    :type y: pd.Series
    :return: Modelo de clasificacion de sklearn
    """
    #Definimos el pipeline
    Titanic_Random_Forest = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf_grid_search",
                GridSearchCV(
                    RandomForestClassifier(),
                    param_grid_,
                    cv=3,
                    scoring="accuracy",
                    return_train_score=True,
                ),
            ),
        ]
    )

    #Ajustamos el modelo
    Titanic_Random_Forest.fit(X, y)

    return Titanic_Random_Forest
