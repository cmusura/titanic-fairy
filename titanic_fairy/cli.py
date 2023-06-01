"""API interfaz de ejecución

"""

import typer

# # from execslotting import logger

"""API interfaz de ejecución de modelo para Titanic.

"""

def main(): 
    from titanic_fairy.helpers.check_data import APP as APP_CHECK

    APP = typer.Typer()
    # chequea los datos y que el path a las tablas sea el correcto
    APP.add_typer(APP_CHECK, name="check_tables")
    # preprocesa los datos y los guarda en una tabla. 
    # APP.add_typer(APP_PREPROCESS, name="preprocess") 
    # # entrena un modelo y lo guarda en result/model
    # APP.add_typer(APP_TRAIN, name="train_model")
    # # genera graficos 
    # APP.add_typer(APP_PREDICT, name="make_graphs")
    # # predice 
    # APP.add_typer(APP_DISTANCE, name="predict")


    
