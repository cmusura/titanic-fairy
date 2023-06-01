# Titanic-Fairy
Este modulo de Python esta diseñado para resolver el clasico problema de Kaggle de clasificar a los osbrevivientes del Titanic.

El proposito del modulo no es resolver el problema a una alta precision, si no mas bien ser un piloto de como construir una libreria de ML que demuestre buenas practicas de codigo. 


# Como usar


## Instalacion 

La libreria esta implementada con poetry, por lo que es necesario tener instalado poetry en tu maquina antes de usar la libreria. 

Luego, una vez dentro de la carpeta principal del proyecto, se debe correr sencillamente:

```
poetry install
poetry shell
```

Una vez dentro de el ambiente virtual la libreria se puede usar dentro de python o usando el CLI. 

## Uso como libreria

Para utilizarlo en Python el script esperado se veria de este modo

```python 
from titanic_fairy.helpers.load_and_save import Load
from titanic_fairy.helpers.train_test_split import stratified_split
from titanic_fairy.metrics.graphics import show_heatmap, check_train_test_split
from titanic_fairy.preprocessing.preprocess import Preprocess
from titanic_fairy.model.train_model import build_model
from titanic_fairy.enums.titanic_fields import Fields, Preprocess
import numpy as np


# Cargamos los datos
titanic_raw_train = Load(train_path = "dataset/train.csv")

# Dividimos en entrenamiento y validacion
raw_train, raw_test = stratified_split(titanic_raw_train)

# Preprocesamos la data
preproc_train = Preprocess().fit_transform(raw_train)
preproc_test = Preprocess().fit_transform(raw_train)

# Separamos en datos y target
X_train = preproc_train.drop([Fields.Survived.value], axis=1)
y_train = preproc_train[Fields.Survived.value]

X_test = preproc_test.drop([Fields.Survived.value], axis=1)
y_test = preproc_test[Fields.Survived.value]

# Armamos el modelo
model = build_model(X_train, y_train)

# Generamos predicciones
model.predict(X_test)
```

Mayores detalles se encuentran en el notebook disponible en el repositorio.

## Uso como CLI 

El paquete titanic-fairy es un CLI estándar hecho con Typer.

### Chequeo de tablas

Se puede chequear que las tablas estan en el formato correcto usando el comando 

```
titanic-fairy check-table --path PATH_DATOS
```

Por default los archivos se encuentran en "dataset\train.csv"

### Preprocesamiento

Se puede usar la bateria de preprocesamiento de la libreria, obteniendo una tabla de datos preprocesados ocupando el comando 

```
titanic-fairy preprocess-table --input-path PATH_DATOS --output-path PATH_SALIDA
```
Ambos parametros estan definidos por default en "dataset\train.csv" y "dataset\preprocess.csv"

### Entrenamiento de modelo

Este comando entrena un modelo con los datos y los guarda en la ruta especificada. 

```
titanic-fairy train-model --input-path PATH_DATOS --output-path PATH_MODELO
```
Ambos parametros estan definidos por default en "dataset\train.csv" y "results\model\titanic_model.joblib"

### Generar predicciones 

Este comando toma un modelo ya entrenado (o bien entrena uno desde 0)
y genera predicciones a partir de un archivo de datos test especificados. 

```
titanic-fairy make-predictions --input-path PATH_DATOS --test-path TEST_PATH --output-path PATH_RESULTADOS --model-path MODEL_PATH --build-model-flag
```
Donde si se entrega --build-model-flag entonces construira el modelo desde 0, en caso contrario usara el entregado en MODEL_PATH