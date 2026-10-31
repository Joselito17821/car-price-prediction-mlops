# Fase 2: scripts y Docker para predecir el precio de autos usados

Proyecto Integrador versión 1, 2026-II. Modelos y Simulación de Sistemas I (profesor Andrés Felipe Parra Barragán).

## Integrantes

- Jose Manuel Londoño Castaño
- Juan José Gómez Castaño

## Problema y relación con la Fase 1

Se mantiene el problema aprobado: predecir `sale_price` (precio de venta en rupias indias) de un auto usado a partir de sus características. El dataset es [Used Car Price Prediction](https://www.kaggle.com/datasets/vrajesh0sharma7/used-car-price-prediction) de Kaggle y el algoritmo sigue siendo un Random Forest entrenado sobre `log(sale_price)`, comparado con un baseline (`DummyRegressor`).

Esta fase convierte el notebook de la Fase 1 (carpeta `../fase-1`, conservada sin cambios) en una aplicación que se ejecuta por línea de comandos y dentro de Docker:

- `train.py` entrena y guarda una versión identificable del modelo.
- `predict.py` carga una versión guardada y predice sobre un CSV nuevo, sin reentrenar.

## Cambios respecto a la Fase 1

- **Código modular.** La lógica del notebook quedó en `src/`: `features.py` (limpieza y variables), `modelo.py` (pipeline, partición y métricas) y `versiones.py` (guardado, carga y metadatos). `train.py` y `predict.py` solo orquestan y usan las mismas funciones, así que no se duplican transformaciones.
- **Limpieza solo en entrenamiento.** Los duplicados, los precios en 0 y las filas sin fecha se eliminan únicamente al entrenar. Al predecir nunca se descarta una fila.
- **`edad` se calcula en ambos scripts** con la misma función, a partir de `ad_created_on`. Por eso el CSV de predicción pide `ad_created_on` y no `edad`.
- **Sin fuga de información.** La partición train/test se hace antes de ajustar la imputación y la codificación, y el preprocesamiento viaja dentro del mismo pipeline guardado.
- **Validación cruzada.** El notebook hacía 5-fold sobre el entrenamiento; `train.py` no la repite. Evalúa baseline y modelo sobre la misma partición de prueba.
- **Cifras de la Fase 1 corregidas.** El texto de la Fase 1 (tabla del notebook y README) reporta MAE 42.956 y RMSE 71.773 para el Random Forest. La salida ejecutada del propio notebook da MAE 43.075 y RMSE 71.891, y es la que reproduce esta fase. En este documento se usan las cifras ejecutadas.
- A la fecha de esta entrega no se había recibido retroalimentación de la Fase 1, por lo tanto lo dejaremos en pendiente.

## Estructura

```
fase-2/
  train.py                  # entrenamiento (punto de entrada)
  predict.py                # predicción (punto de entrada)
  src/                      # lógica compartida
  tests/test_aplicacion.py  # pruebas con pytest
  conftest.py               # permite a pytest importar los scripts
  data/
    Used_Car_Price_Prediction.csv   # dataset completo
    ejemplo_prediccion.csv          # 10 filas de ejemplo, sin sale_price
  models/
    v1/  v2/                # versiones: modelo.joblib, metrics.json, metadata.json
  outputs/                  # resultados de las predicciones
  Dockerfile  .dockerignore  requirements.txt
```

## Entorno e instalación

- **Python 3.14** (probado con 3.14.3 en local y 3.14.10 en la imagen Docker).
- Terminal usada en este documento: **PowerShell (Windows)**.

Desde la carpeta `fase-2/`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Mac o Linux la activación es `source .venv/bin/activate`. Todas las dependencias tienen versión exacta en `requirements.txt` (incluido pytest), y los modelos guardados solo se cargan con esas versiones.

Si tienes varias versiones de Python instaladas, crea el entorno indicando la 3.14: `py -3.14 -m venv .venv` o `python3.14 -m venv .venv`, según cómo se haya instalado Python. Con el entorno activado, `python --version` debe mostrar 3.14.x.

## Datos

El dataset completo ya está en el repositorio: `fase-2/data/Used_Car_Price_Prediction.csv` (7.400 registros y 29 columnas; fuente original en Kaggle, enlace arriba). No hace falta descargar nada ni usar credenciales. Tras la limpieza de entrenamiento quedan 7.395 filas.

## Entrenamiento

Desde `fase-2/`, en una sola línea:

```powershell
python train.py --data data/Used_Car_Price_Prediction.csv --target sale_price --output-dir models --version v3 --seed 42 --n-estimators 300
```

| Argumento | Descripción | Por defecto |
|---|---|---|
| `--data` | CSV de datos etiquetados | (obligatorio) |
| `--target` | Variable objetivo | `sale_price` |
| `--output-dir` | Carpeta donde se guardan las versiones | `models` |
| `--version` | Identificador de la versión, por ejemplo `v3` | (obligatorio) |
| `--seed` | Semilla aleatoria (partición y modelo) | `42` |
| `--test-size` | Fracción de datos para prueba | `0.2` |
| `--n-estimators` | Árboles del Random Forest | `300` |

Para ver las opciones: `python train.py --help`. Si la versión ya existe, el script termina con error sin tocar nada: hay que usar otro identificador. Con 300 árboles tarda un par de minutos.

## Predicción

```powershell
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

`--model` acepta la ruta de `modelo.joblib` o la carpeta de la versión (`models/v1`), así se elige explícitamente la versión a usar. Ayuda: `python predict.py --help`.

### Esquema del CSV de entrada

El CSV debe tener estas 15 columnas (el orden no importa y `sale_price` no hace falta):

| Columna | Tipo | Nota |
|---|---|---|
| `yr_mfr` | numérica | año de fabricación |
| `kms_run` | numérica | kilómetros recorridos |
| `total_owners` | numérica | número de dueños |
| `warranty_avail` | numérica (True/False) | garantía disponible |
| `ad_created_on` | fecha ISO 8601 | de aquí se calcula `edad` |
| `fuel_type`, `city`, `body_type`, `transmission`, `source`, `car_rating`, `fitness_certificate`, `make`, `model`, `registered_state` | categórica | texto |

Reglas de comportamiento:

- **Columnas adicionales:** se ignoran.
- **Valores nulos:** son válidos; el pipeline guardado los imputa con lo aprendido en el entrenamiento (mediana en numéricas, valor más frecuente en categóricas). Una celda vacía no equivale a una columna ausente.
- **Categorías no vistas en el entrenamiento:** se codifican como ceros, sin error.
- **Orden y filas:** se conserva el orden y se genera una predicción por fila, sin eliminar ninguna.
- **Salida:** el mismo CSV de entrada más una columna `prediccion` (rupias indias).
- **Errores previsibles** (columna ausente, tipo incompatible, modelo o archivo inexistente): mensaje claro que empieza con `Error:`, código de salida `1` y sin archivo de resultados.

## Versiones del modelo

Cada ejecución de `train.py` crea `models/<versión>/` con:

- `modelo.joblib`: pipeline completo (preprocesamiento y modelo).
- `metrics.json`: MAE, RMSE y R² del modelo y del baseline.
- `metadata.json`: versión, fecha, tipo de problema, objetivo, algoritmo y parámetros, semilla, columnas de entrada y tipos, datos usados (fuente, archivo, observaciones), partición, métricas, versión de Python y hash del commit.

El repositorio incluye dos versiones entrenadas con este mismo código:

| Versión | Parámetros |
|---|---|
| `v1` | 300 árboles, semilla 42 |
| `v2` | 200 árboles, semilla 7 |

Los resultados de predicción se guardan en `outputs/` (ignorado por Git).

## Docker

La imagen (`python:3.14-slim`) contiene el código y las dependencias, pero **no entrena al construirse**. Los datos, modelos y resultados se montan como carpetas para que persistan fuera del contenedor. Desde `fase-2/`, en PowerShell y en una sola línea cada comando:

```powershell
docker build -t modelos-fase2:1.0 .
```

Entrenar dentro del contenedor:

```powershell
docker run --rm -v "${PWD}\data:/app/data:ro" -v "${PWD}\models:/app/models" -e GIT_COMMIT=$(git rev-parse HEAD) modelos-fase2:1.0 python train.py --data data/Used_Car_Price_Prediction.csv --target sale_price --output-dir models --version v-docker --seed 42 --n-estimators 10
```

Predecir dentro del contenedor:

```powershell
docker run --rm -v "${PWD}\data:/app/data:ro" -v "${PWD}\models:/app/models:ro" -v "${PWD}\outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/pred_docker.csv
```

- `-e GIT_COMMIT=...` es opcional: dentro de la imagen no hay `git`, así que el hash del commit en `metadata.json` solo se registra si se pasa así.
- En Bash, los montajes se escriben `-v "$PWD/data:/app/data:ro"` (y igual con `models` y `outputs`).
- La versión `v-docker` no está en el repositorio; se genera al ejecutar el comando.

## Pruebas

Desde `fase-2/`:

```powershell
python -m pytest
```

Son 5 pruebas en `tests/test_aplicacion.py`, con datos sintéticos pequeños y directorios temporales (no tocan `models/` ni `data/`, y no necesitan internet):

1. **Entrenamiento y almacenamiento:** se generan modelo cargable, métricas (modelo y baseline) y metadatos.
2. **Carga y predicción:** el modelo guardado da una predicción por fila y los mismos resultados que antes de guardarlo (con tolerancia numérica).
3. **Valores faltantes:** una entrada con nulos válidos se procesa y no se pierde ninguna fila.
4. **Entrada inválida:** si falta una columna hay un error claro, `predict.py` termina con código distinto de cero y no crea el archivo de salida.
5. **Versionamiento:** una segunda versión conserva la primera y un identificador repetido no sobrescribe archivos.

## Resultados

Métricas sobre el conjunto de prueba (20 % de los datos), en rupias indias:

| Versión | Modelo MAE | Modelo RMSE | Modelo R² | Baseline MAE | Baseline RMSE | Baseline R² |
|---|---|---|---|---|---|---|
| `v1` | 43.075 | 71.891 | 0,930 | 179.222 | 278.571 | -0,055 |
| `v2` | 42.804 | 76.200 | 0,922 | 179.123 | 283.491 | -0,077 |

- El Random Forest reduce el MAE en cerca de un 76 % frente al baseline en ambas versiones.
- `v1` reproduce la salida ejecutada del notebook de la Fase 1 (ver la sección de cambios).
- Cada versión usa su semilla también para la partición, así que `v1` y `v2` se evalúan sobre conjuntos de prueba distintos y sus métricas no son estrictamente comparables entre sí. La especificación no exige que una mejore a la otra.
- Con las mismas versiones de librerías, los resultados son prácticamente iguales entre computadores, aunque no siempre idénticos. Con 10 árboles y semilla 42 se obtuvo MAE 46.039 en local y en un primer contenedor Docker, y MAE 45.815 dentro de Docker en otro computador (Linux x86_64, mismas versiones de librerías). El R² fue 0,920 en todos los casos. La diferencia (cerca de 0,5 % del MAE) es numérica y propia de cada plataforma.

## Limitaciones conocidas

- El modelo se entrenó con anuncios de autos usados de la India y sus precios están en rupias; no es válido para otros mercados ni para anuncios muy alejados en el tiempo.
- La partición entrenamiento/prueba es aleatoria por fila, no agrupada. Se comprobó si un mismo auto aparece publicado más de una vez: en el dataset solo 1 fila (0,01 % del total) repite todas las variables de entrada de otra fila sin contar la fecha del anuncio, y es un duplicado exacto que la limpieza de entrenamiento elimina. Tras la limpieza no queda ninguna, así que no hay copias del mismo anuncio repartidas entre entrenamiento y prueba. Con datos nuevos en que un auto se publique varias veces convendría una partición agrupada.
- `edad` depende de la fecha del anuncio (`ad_created_on`), que es obligatoria en el CSV de predicción.
- El Random Forest tiene componentes aleatorios: los resultados son reproducibles solo con la misma semilla, los mismos datos y las mismas versiones de librerías.
- Las categorías nuevas (por ejemplo una marca no vista) no aportan información al modelo: se codifican como ceros.
- Un valor con tipo incompatible (por ejemplo texto en `kms_run`) detiene la predicción completa en lugar de predecir las demás filas.
- Cada `modelo.joblib` pesa entre 25 y 40 MB.
- No hay API REST ni monitoreo: corresponden a fases siguientes.

## Evidencia de ejecución

Entrenamiento de `v1`:

```
Versión 'v1' entrenada.
  baseline: MAE=179222  RMSE=278571  R2=-0.055
  modelo: MAE=43075  RMSE=71891  R2=0.930
Archivos generados:
  models\v1\modelo.joblib
  models\v1\metrics.json
  models\v1\metadata.json
```

Predicción con `v1`:

```
10 predicciones generadas con la versión 'v1'.
Resultados guardados en: outputs/predicciones_v1.csv
```

Error previsible (modelo inexistente, código de salida `1`):

```
Error: No se encontró el modelo en models\nada
```

Docker: la imagen se construyó correctamente, y el entrenamiento y la predicción dentro del contenedor terminaron bien. Los archivos de `models/` y `outputs/` quedaron en el computador anfitrión después de eliminar el contenedor.

```
Versión 'v-docker' entrenada.
  baseline: MAE=179222  RMSE=278571  R2=-0.055
  modelo: MAE=46039  RMSE=76852  R2=0.920
...
10 predicciones generadas con la versión 'v-docker'.
Resultados guardados en: outputs/pred_docker.csv
```

Pruebas:

```
================================================= test session starts =================================================
platform win32 -- Python 3.14.10, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\USUARIO\OneDrive\Desktop\Universidad\2026-2\Modelos y simulacion de sistemas\Proyecto Modelos\car-price-prediction-mlops\fase-2
collected 5 items

tests\test_aplicacion.py .....                                                                                   [100%]

================================================= 5 passed in 10.10s ==================================================
```

## Entrega

- Etiqueta de Git de la Entrega 2: **`entrega-2`**, sobre la rama `main`.
- Repositorio: https://github.com/Joselito17821/car-price-prediction-mlops

## Fuentes consultadas

- Dataset: [Used Car Price Prediction](https://www.kaggle.com/datasets/vrajesh0sharma7/used-car-price-prediction), Kaggle (vrajesh0sharma7).
- Documentación de scikit-learn (Pipeline, ColumnTransformer, TransformedTargetRegressor, RandomForestRegressor), pandas, joblib, pytest y Docker.
- Especificación técnica de la Fase 2 del curso.
