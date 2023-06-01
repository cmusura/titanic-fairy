"""API interfaz de ejecución

"""
from pathlib import Path
import typer
import pandas as pd
from titanic_fairy.helpers.check_data import check_table_
from titanic_fairy.enums.titanic_fields import Fields, Preprocess_
from titanic_fairy.preprocessing.preprocess import Preprocess

# # from execslotting import logger

"""API interfaz de ejecución de modelo para Titanic.

"""

app = typer.Typer()


TRAIN_PATH_OPTION = typer.Option(
    str(Path("dataset/train.csv")),
    help="Ruta al archivo de datos para ser ingestado por el modelo",
)

INPUT_PREPROCESS_PATH_OPTION = typer.Option(
    str(Path("dataset/train.csv")),
    help="Ruta al archivo de datos para ser ingestado por el modelo",
)

OUTPUT_PREPROCESS_PATH_OPTION = typer.Option(
    str(Path("dataset/preprocess.csv")),
    help="Ruta al archivo de datos para ser ingestado por el modelo",
)


@app.command()
def check_table(path: Path = TRAIN_PATH_OPTION):
    """Chequea que la tabla este lista apra ser procesada

    :param path: Ruta al archivo, defaults to PATH_OPTION
    :type path: Path, optional
    """
    check = check_table_(path)
    if check:
        typer.echo("La tabla esta lsita para procesarse.")


@app.command()
def preprocess_table(
    input_path: Path = TRAIN_PATH_OPTION,
    output_path: Path = TRAIN_PATH_OPTION,
):
    """
    Preprocesa los datos en input_path y guarda los datos preprocesados en output_path

    :param input_path: Ruta a tabla de Titanic, defaults to TRAIN_PATH_OPTION
    :type input_path: Path, optional
    :param output_path: Ruta donde se desea guardar el preprocesamiento, defaults to TRAIN_PATH_OPTION
    :type output_path: Path, optional
    """

    df = pd.read_csv(input_path)
    typer.echo(len(df))
    preproc = Preprocess().fit_transform(df)
    preproc.to_csv(output_path)
    typer.echo("Done.")
    return None


# def main():
#     from titanic_fairy.helpers.check_data import APP as APP_CHECK

#     APP = typer.Typer()
#     # chequea los datos y que el path a las tablas sea el correcto
#     APP.add_typer(APP_CHECK, name="check-tables")
#     # preprocesa los datos y los guarda en una tabla.
#     # APP.add_typer(APP_PREPROCESS, name="preprocess")
#     # # entrena un modelo y lo guarda en result/model
#     # APP.add_typer(APP_TRAIN, name="train_model")
#     # # genera graficos
#     # APP.add_typer(APP_PREDICT, name="make_graphs")
#     # # predice
#     # APP.add_typer(APP_DISTANCE, name="predict")
#     APP()
