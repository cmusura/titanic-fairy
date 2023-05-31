"""Modulo que guarda los nombres de la base de datos original de Titanic """
from enum import Enum


class Fields(Enum):
    """Guardamos los campos por default de los datos de Titanic.

    Esto evita errores de tipeo y evita problema si el nombre de un campo cambia de nombre
    """

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
