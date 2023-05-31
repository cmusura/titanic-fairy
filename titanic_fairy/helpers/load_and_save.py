"""Modulo de lectura de los datos"""

import pandas as pd


def Load(train_path="dataset//train.csv", test_path="dataset//test.csv", train=True):
    """Lee los datos de entrenamiento/testeo del problema Titanic

    Se considera que los datos se enceuntran en la carpeta dataset bajo el nombre de train.csv y test.csv.
    Por default lee los datos de entrenamiento.

    :param train_path: Path a datos de entrenamiento del modelo, defaults to 'dataset/train.csv'
    :type train_path: str, optional
    :param test_path: Path a datos de testeo del modelo, defaults to 'dataset/test.csv'
    :type test_path: str, optional
    :param train: Indicar si leemos datos de entrenamiento o de test, defaults to True
    :type train: bool, optional
    """
    if train:
        df = pd.read_csv(train_path)
    else:
        df = pd.read_csv(test_path)
    return df
