"""API interfaz de ejecución

"""
from pathlib import Path
import typer
from titanic_fairy.helpers.check_data import check_table_

# # from execslotting import logger

"""API interfaz de ejecución de modelo para Titanic.

"""

app = typer.Typer()


TRAIN_PATH_OPTION = typer.Option(
    str(Path("dataset/train.csv"))
    , help="Ruta al archivo de datos para ser ingestado por el modelo"
)

@app.command()
def check_table(path : Path = TRAIN_PATH_OPTION):
    """Chequea que la tabla este lista apra ser procesada

    :param path: Ruta al archivo, defaults to PATH_OPTION
    :type path: Path, optional
    """
    check = check_table_(path)
    if check:
        typer.echo("La tabla esta lsita para procesarse.")



@app.command()
def shoot():
    """Chequea que la tabla este lista apra ser procesada

    :param path: Ruta al archivo, defaults to PATH_OPTION
    :type path: Path, optional
    """

    typer.echo("pium")



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


    
