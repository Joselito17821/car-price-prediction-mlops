"""Limpieza y construcción de variables, compartidas por train.py y predict.py."""
import pandas as pd

TARGET = "sale_price"
COL_FECHA = "ad_created_on"
NUM_COLS = ["yr_mfr", "kms_run", "total_owners", "edad", "warranty_avail"]
CAT_COLS = ["fuel_type", "city", "body_type", "transmission", "source",
            "car_rating", "fitness_certificate", "make", "model", "registered_state"]
FEATURES = NUM_COLS + CAT_COLS
# Columnas que debe traer el CSV (edad se calcula desde ad_created_on)
COLUMNAS_ENTRADA = ["yr_mfr", "kms_run", "total_owners", "warranty_avail",
                    COL_FECHA] + CAT_COLS


def limpiar_entrenamiento(df, target=TARGET):
    """Filtros solo para entrenamiento; nunca se aplican a datos nuevos."""
    df = df.drop_duplicates()
    df = df[df[target] > 0]
    df = df.dropna(subset=[COL_FECHA])
    return df.reset_index(drop=True)


def crear_features(df):
    """Calcula edad y deja solo las columnas del modelo. No elimina filas."""
    df = df.copy()
    fecha = pd.to_datetime(df[COL_FECHA], format="ISO8601", errors="coerce")
    df["edad"] = fecha.dt.year - df["yr_mfr"]
    return df[FEATURES]