"""Modulo que guarda los nombres de la base de datos original de Titanic. """
from enum import Enum


class Fields(Enum):
    """Nombre de los campos de los datos de Titanic."""

    ID = "PassengerId"
    Survived = "Survived"
    Pclass = "Pclass"
    Name = "Name"
    Sex = "Sex"
    Age = "Age"
    Sibsp = "SibSp"
    Parch = "Parch"
    Ticket = "Ticket"
    Fare = "Fare"
    Cabin = "Cabin"
    Embarked = "Embarked"


class Preprocess_(Enum):
    """Variables consideradas en cada paso del procesamiento

    Parametros:
        Impute (Dict): Requiere un diccionario que incluya las variables a imputar como llave y la estrategia como valor
        Encode (List): Lista de variables categoricas a codificar
        Drop (List): Lista de variables a desechar (Incluye las que se codifican por defecto)
    """

    Impute = {
        Fields.Age.value: "mean",
        Fields.Cabin.value: "most_frequent",
        Fields.Embarked.value: "most_frequent",
        Fields.Fare.value: "mean",
    }

    Encode = [Fields.Embarked.value, Fields.Sex.value]

    # Desechamos el ID, el nombre y el ticket al igual que las columnas que estan codificadas por el modulo encoder
    Drop = [
        Fields.ID.value,
        Fields.Name.value,
        Fields.Ticket.value,
        Fields.Cabin.value,
    ] + [Fields.Embarked.value, Fields.Sex.value]

    # Criterios en base a los cuales realizar la separacion estratificada
    Train_Test_Criteria = [
        Fields.Survived.value,
        Fields.Pclass.value,
        Fields.Sex.value,
    ]


class Model_Fields(Enum):
    """Contiene hiperparametros de el modelo"""

    Param_Grid = {
        "n_estimators": [10, 100, 200, 500],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 3, 4],
    }
