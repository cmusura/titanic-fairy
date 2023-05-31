"""Test de que nombres de las columnas son los correctos"""
import pytest
import pandas as pd
import os
from titanic_fairy.enums.titanic_fields import Fields


def test_enums():
    """Chequeamos que los campos de los datos a ocupar corresponden a los valores esperados por la libreria"""

    # Estos son los valores esperados
    fields_names = [field.value for field in Fields]

    # Recorremos cada tabla
    for csv in os.listdir("dataset"):
        dataset = pd.read_csv("dataset/" + csv)
        # Para cada columna chqueamos si corresponde a algun valor esperado
        for col in dataset.columns:
            assert col in fields_names
