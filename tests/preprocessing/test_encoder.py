import pytest
from titanic_fairy.preprocessing.encoder import FeatureEncoder
from titanic_fairy.enums.titanic_fields import Fields
import pandas as pd

# Definimos una tabla auxiliar de prueba
TEST_DATA = [
    pd.DataFrame(
        {
            Fields.Age.value: [1, 2, 1, 2, 4, 1, 2],
            Fields.Embarked.value: ["z", "z", "z", "w", "x", "w", "z"],
            Fields.Sex.value: ["a", "b", "b", "b", "a", "a", "b"],
            Fields.Fare.value: [8, 8, 8, 8, 8, 8, 8],
        }
    )
]

# Este es el resultado esperado tras imputar datos faltantes
EXPECTED = [
    pd.DataFrame(
        {
            Fields.Age.value: [1, 2, 1, 2, 4, 1, 2],
            Fields.Embarked.value: ["z", "z", "z", "w", "x", "w", "z"],
            Fields.Sex.value: ["a", "b", "b", "b", "a", "a", "b"],
            Fields.Fare.value: [8, 8, 8, 8, 8, 8, 8],
            "w": [0, 0, 0, 1, 0, 1, 0],
            "z": [1, 1, 1, 0, 0, 0, 1],
            "x": [0, 0, 0, 0, 1, 0, 0],
            "a": [1, 0, 0, 0, 1, 1, 0],
            "b": [0, 1, 1, 1, 0, 0, 1],
        }
    )
]


# Con este decorador despues es facil agregar mas tablas de prueba
@pytest.mark.parametrize("test_data, expected", zip(TEST_DATA, EXPECTED))
def test_model_params(test_data, expected):
    test_data = FeatureEncoder().fit_transform(test_data)
    assert (test_data["w"] == expected["w"]).all()
    assert (test_data["z"] == expected["z"]).all()
    assert (test_data["x"] == expected["x"]).all()
    assert (test_data["a"] == expected["a"]).all()
    assert (test_data["b"] == expected["b"]).all()
