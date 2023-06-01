"""Test de imputadores de valores"""
from titanic_fairy.preprocessing.dropper import FeatureDropper
from titanic_fairy.enums.titanic_fields import Fields, Preprocess
import pytest
import pandas as pd
import numpy as np

# Definimos una tabla auxiliar de prueba
TEST_DATA = [
    pd.DataFrame(
        {
            Fields.Age.value: [1, 2, 1, 2, np.nan, 1, 2],
            Fields.Name.value: [1, 2, 1, 2, np.nan, 1, 2],
            Fields.ID.value: [1, 2, 1, 2, np.nan, 1, 2],
            Fields.Cabin.value: [1, 2, 1, 2, np.nan, 1, 2],
            Fields.Ticket.value: [1, 2, 1, 2, np.nan, 1, 2],
            Fields.Sex.value: ["a", "a", "a", "b", np.nan, "a", "b"],
            Fields.Embarked.value: ["a", "b", "b", "b", np.nan, "a", "b"],
            Fields.Fare.value: [8, 8, 8, 8, np.nan, 8, 8],
        }
    )
]

# Este es el resultado esperado tras imputar datos faltantes
EXPECTED = [
    pd.DataFrame(
        {
            Fields.Age.value: [1, 2, 1, 2, np.nan, 1, 2],
            Fields.Fare.value: [8, 8, 8, 8, np.nan, 8, 8],
        }
    )
]


# Con este decorador despues es facil agregar mas tablas de prueba
@pytest.mark.parametrize("test_data, expected", zip(TEST_DATA, EXPECTED))
def test_model_params(test_data, expected):
    test_data = FeatureDropper().fit_transform(test_data)
    for i in range(len(test_data.columns)):
        assert test_data.columns[i] in expected.columns
