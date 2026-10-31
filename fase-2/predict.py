"""Genera predicciones con una versión guardada del modelo (sin reentrenar)."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.features import crear_features
from src.versiones import cargar_version


def leer_entrada(ruta):
    """Lee el CSV de entrada y verifica que exista y tenga filas."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {ruta}")
    try:
        df = pd.read_csv(ruta)
    except ValueError as e:
        raise ValueError(f"No se pudo leer el CSV {ruta}: {e}") from e
    if df.empty:
        raise ValueError(f"El archivo {ruta} no tiene filas para predecir")
    return df


def validar_entrada(df, esquema):
    """Verifica columnas requeridas y tipos compatibles. Un nulo es válido."""
    if not esquema:
        raise ValueError("La versión del modelo no define 'columnas_entrada' en metadata.json")
    faltantes = [c for c in esquema if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")

    invalidas = []
    for col, tipo in esquema.items():
        serie = df[col]
        if tipo == "numerica":
            convertida = pd.to_numeric(serie, errors="coerce")
        elif tipo == "fecha":
            convertida = pd.to_datetime(serie, format="ISO8601", errors="coerce")
        else:
            continue
        if (convertida.isna() & serie.notna()).any():
            invalidas.append(f"{col} (se esperaba tipo {tipo})")
    if invalidas:
        raise ValueError("Valores con tipo incompatible en: " + ", ".join(invalidas))


def predecir(modelo, df):
    """Aplica las mismas transformaciones del entrenamiento y devuelve una predicción por fila, en orden."""
    try:
        return modelo.predict(crear_features(df))
    except Exception as e:
        raise ValueError(f"No se pudieron generar las predicciones con estos datos: {e}") from e


def guardar_salida(df, predicciones, ruta):
    """Escribe el CSV con una columna 'prediccion', sin dejar archivos a medias."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida = df.copy()
    salida["prediccion"] = predicciones
    temporal = ruta.with_name(ruta.name + ".tmp")
    salida.to_csv(temporal, index=False)
    temporal.replace(ruta)


def parsear_argumentos():
    p = argparse.ArgumentParser(
        description="Genera predicciones con una versión guardada del modelo.")
    p.add_argument("--model", required=True,
                   help="Ruta de modelo.joblib o de la carpeta de la versión, ej. models/v1")
    p.add_argument("--input", required=True,
                   help="CSV con las variables de entrada (sin variable objetivo)")
    p.add_argument("--output", required=True, help="CSV de salida con la columna 'prediccion'")
    return p.parse_args()


def main():
    args = parsear_argumentos()
    try:
        modelo, metadata = cargar_version(args.model)
        df = leer_entrada(args.input)
        validar_entrada(df, metadata.get("columnas_entrada"))
        predicciones = predecir(modelo, df)
        guardar_salida(df, predicciones, args.output)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"{len(df)} predicciones generadas con la versión '{metadata['version']}'.")
    print(f"Resultados guardados en: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())