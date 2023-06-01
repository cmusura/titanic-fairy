"""API interfaz de ejecución

"""

import typer

# # from execslotting import logger

"""API interfaz de ejecución de modelo para Titanic.

"""

def main(): 

    from titanic_fairy.helpers.load_and_save import Load
    from titanic_fairy.helpers.train_test_split import stratified_split
    from titanic_fairy.metrics.graphics import show_heatmap, check_train_test_split
    from titanic_fairy.preprocessing.preprocess import Preprocess
    from titanic_fairy.model.train_model import build_model
    from titanic_fairy.enums.titanic_fields import Fields, Preprocess
    import numpy as np


    # Cargamos los datos
    titanic_raw_train = Load(train_path = "dataset/train.csv")

    # Dividimos en entrenamiento y validacion
    raw_train, raw_test = stratified_split(titanic_raw_train)

    # Preprocesamos la data
    preproc_train = Preprocess().fit_transform(raw_train)
    preproc_test = Preprocess().fit_transform(raw_train)

    # Separamos en datos y target
    X_train = preproc_train.drop([Fields.Survived.value], axis=1)
    y_train = preproc_train[Fields.Survived.value]

    X_test = preproc_test.drop([Fields.Survived.value], axis=1)
    y_test = preproc_test[Fields.Survived.value]

    # Armamos el modelo
    model = build_model(X_train, y_train)

    # Generamos predicciones
    model.predict(X_test)


