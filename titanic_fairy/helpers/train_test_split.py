"""Modulo auxiliar para separar datos de entrenamiento y validacion

Se ocupa la separacion estratificada para obtener datos de validacion balanceados y 
no tener problemas con el sesgo de seleccion en el problema de Titanic

"""

from sklearn.model_selection import StratifiedShuffleSplit
from titanic_fairy.enums.titanic_fields import Fields, Preprocess_
import pandas as pd


def stratified_split(df: pd.DataFrame, n_splits=1, test_size=0.2):
    """Realiza la separacion estratificada de los datos de entrenamiento.

    La estratificacion mantiene las clases balanceadas. El criterio de balanceo esta definido en el modulo Enum.

    """
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    for train_indices, test_indices in split.split(
        df, df[Preprocess_.Train_Test_Criteria.value]
    ):
        train_set = df.loc[train_indices]
        test_set = df.loc[test_indices]

    return train_set, test_set
