"""Modulo que implementa distintos graficos de interes"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from titanic_fairy.enums.titanic_fields import Fields


def show_heatmap(df: pd.DataFrame, path: str = None, save=False):
    """Genera un mapa de calor con la correlacion de las entradas de un df

    :param df: DataFrame de los datos a graficar
    :type df: pd.DataFrame
    :param path: Path a donde guardar la imagen
    :type path: str
    :param save: Si guardar o no una imagen del grafico, defaults to False
    :type save: bool, optional
    """
    heatmap = sns.heatmap(df.corr(numeric_only=True), cmap="YlGnBu")
    if save:
        if path is None:
            raise Exception("Si save es True, se debe entregar un path")
        heatmap.savefig(path)
    plt.show()


def check_train_test_split(train_set: pd.DataFrame, test_set: pd.DataFrame):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Train")
    train_set[Fields.Survived.value].hist(label=Fields.Survived.value)
    train_set[Fields.Pclass.value].hist(label=Fields.Pclass.value)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Validation")
    test_set[Fields.Survived.value].hist(label=Fields.Survived.value)
    test_set[Fields.Pclass.value].hist(label=Fields.Pclass.value)
    plt.legend()

    plt.show()
