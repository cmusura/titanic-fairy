""" Modulo para chequear que las tablas de Titanic esten en el path definido"""

from pathlib import Path
import pandas as pd
from titanic_fairy.enums.titanic_fields import Fields



def check_table_(data_file: Path):
    """Chequea que las tablas en el path entregado esten parseadas adecuadamente
    :param data_file: _description_
    :type data_file: Path
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
    return flag


