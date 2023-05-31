"""Modulo que guarda los nombres de la base de datos original de Titanic """
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

class Fields_Cast(Enum):
    """Campos por default de los datos de Titanic."""

    ID = int
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