""" Modulo para chequear que las tablas de Titanic esten en el path definido"""

from pathlib import Path
import pandas as pd
import typer
from titanic_fairy.enums.titanic_fields import Fields

APP = typer.Typer()


PATH_OPTION = typer.Option(
    str(Path("dataset/", "train.csv"))
    , help="Ruta al archivo de datos para ser ingestado por el modelo"
)


@APP.command()
def check_table(data_file: Path = PATH_OPTION):
    """Chequea que las tablas en el path entregado esten parseadas adecuadamente
    :param data_file: _description_, defaults to PATH_OPTION
    :type data_file: Path, optional
    """
    fields_names = [field.value for field in Fields]
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print("Archivo no encontrado. Descargue datos de Kaggle y coloquelos en directorio dataset")
    flag = True
    for col in df.columns:
        if col not in fields_names:
            raise ValueError("Datos poseen columna no esperada")
        flag = False
    if flag:
        print("Datos se encuentran en orden!")


