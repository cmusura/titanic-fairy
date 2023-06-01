"""Test de imputadores de valores"""
from titanic_fairy.preprocessing.imputer import Imputer
from titanic_fairy.enums.titanic_fields import Fields
import pytest
import pandas as pd
import numpy as np

# Definimos una tabla auxiliar de prueba
TEST_DATA = [
    pd.DataFrame(
        {
            Fields.Age.value: [1, 2, 1, 2, np.nan, 1, 2],
            Fields.Cabin.value: ["a", "a", "a", "b", np.nan, "a", "b"],
            Fields.Embarked.value: ["a", "b", "b", "b", np.nan, "a", "b"],
            Fields.Fare.value: [8, 8, 8, 8, np.nan, 8, 8],
        }
    )
]

# Este es el resultado esperado tras imputar datos faltantes
EXPECTED = [
    pd.DataFrame(
        {
            Fields.Age.value: [1, 2, 1, 2, 1.5, 1, 2],
            Fields.Cabin.value: ["a", "a", "a", "b", "a", "a", "b"],
            Fields.Embarked.value: ["a", "b", "b", "b", "b", "a", "b"],
            Fields.Fare.value: [8, 8, 8, 8, 8, 8, 8],
        }
    )
]


# Con este decorador despues es facil agregar mas tablas de prueba
@pytest.mark.parametrize("test_data, expected", zip(TEST_DATA, EXPECTED))
def test_model_params(test_data, expected):
    test_data = Imputer().fit_transform(test_data)
    assert (test_data[Fields.Age.value] == expected[Fields.Age.value]).all()
    assert (test_data[Fields.Cabin.value] == expected[Fields.Cabin.value]).all()
    assert (test_data[Fields.Embarked.value] == expected[Fields.Embarked.value]).all()
    assert (test_data[Fields.Fare.value] == expected[Fields.Fare.value]).all()
