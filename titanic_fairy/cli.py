"""API interfaz de ejecución

"""
from pathlib import Path
import typer
import pandas as pd
from titanic_fairy.helpers.check_data import check_table_
from titanic_fairy.enums.titanic_fields import Fields, Preprocess_
from titanic_fairy.preprocessing.preprocess import Preprocess
from titanic_fairy.model.train_model import build_model
import joblib


"""

Implementacion del CLI para modelo Titanic

"""

app = typer.Typer()

# Definimos las opciones por defecto
TRAIN_PATH_OPTION = typer.Option(
    str(Path("dataset/train.csv")),
    help="Ruta al archivo de datos para ser ingestado por el modelo",
)

TEST_PATH_OPTION = typer.Option(
    str(Path("dataset/test.csv")),
    help="Ruta al archivo de datos test para realizar la prediccion",
)

PREPROCESS_PATH_OPTION = typer.Option(
    str(Path("dataset/preprocess.csv")),
    help="Ruta de salida de la tabla preprocesada",
)

MODEL_PATH_OPTION = typer.Option(
    str(Path("results/model/titanic_model.joblib'=")),
    help="Ruta del modelo a ser entrenado/utilizado",
)

IMGS_PATH_OPTION = typer.Option(
    str(Path("results/imgs")),
    help="Directorio donde guardar imagenes",
)

PREDICTIONS_PATH_OPTION = typer.Option(
    str(Path("results/predictions.csv")),
    help="Ruta de salida de predicciones realizadas por el modelo",
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
    return None


@app.command()
def preprocess_table(
    input_path: Path = TRAIN_PATH_OPTION,
    output_path: Path = PREPROCESS_PATH_OPTION,
):
    """
    Preprocesa los datos en input_path y guarda los datos preprocesados en output_path

    :param input_path: Ruta a tabla de Titanic, defaults to TRAIN_PATH_OPTION
    :type input_path: Path, optional
    :param output_path: Ruta donde se desea guardar el preprocesamiento, defaults to TRAIN_PATH_OPTION
    :type output_path: Path, optional
    """

    df = pd.read_csv(input_path)
    preproc = Preprocess().fit_transform(df)
    preproc.to_csv(output_path)
    typer.echo("Done.")
    return None


@app.command()
def train_model(
    input_path: Path = TRAIN_PATH_OPTION, output_path: Path = MODEL_PATH_OPTION
):
    """
    Realiza el pipeline completo, desde preprocesamiento
    a salida de un modelo. El modelo queda en formato joblib en la ruta output_path
    """

    df = pd.read_csv(input_path)
    preproc = Preprocess().fit_transform(df)
    X = preproc.drop([Fields.Survived.value], axis=1)
    y = preproc[Fields.Survived.value]
    model = build_model(X, y)
    joblib.dump(model, output_path)
    typer.echo("Done.")
    return None


@app.command()
def make_predictions(
    input_path: Path = TRAIN_PATH_OPTION,
    test_path: Path = TEST_PATH_OPTION,
    output_path: Path = PREDICTIONS_PATH_OPTION,
    build_model_flag: bool = True,
):
    """
    Toma o entrena un modelo (dependiendo del flag build_model) y realiza las predicciones correspondientes
    y las entrega como resultados en la ruta indicada

    """
    if build_model_flag:
        df = pd.read_csv(input_path)
        preproc = Preprocess().fit_transform(df)
        X = preproc.drop([Fields.Survived.value], axis=1)
        y = preproc[Fields.Survived.value]
        model = build_model(X, y)
    else:
        try:
            model = joblib.load(MODEL_PATH_OPTION)
        except ModuleNotFoundError:
            typer.echo(
                "Si build_model_flag es False, se debe entregar una ruta a un modelo en formato joblib."
            )

    # Leemos los datos test
    test = pd.read_csv(test_path)

    # Guardamos los ID de los pasajeros a predecir
    passenger_id = test[Fields.ID.value]

    # Aplicamos el pipeline
    test = Preprocess().fit_transform(test)
    results = model.predict(test)

    # Formateamos y guardamos los resultados
    output = pd.DataFrame(
        {Fields.ID.value: passenger_id, Fields.Survived.value: results}
    )

    output.to_csv(output_path)

    typer.echo("Done.")
    return None
