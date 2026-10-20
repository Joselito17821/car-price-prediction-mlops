"""Entrena el modelo de precios de autos usados y guarda una versión en models/<version>/."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.features import COLUMNAS_ENTRADA, TARGET, crear_features, limpiar_entrenamiento
from src.modelo import construir_baseline, construir_modelo, dividir_datos, evaluar
from src.versiones import armar_metadata, guardar_version


def leer_datos(ruta, target):
    """Lee el CSV y verifica que existan el objetivo y las columnas necesarias."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")
    df = pd.read_csv(ruta)
    faltantes = [c for c in [target] + COLUMNAS_ENTRADA if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")
    return df


def entrenar(ruta_datos, target, output_dir, version, seed, test_size, n_estimators):
    if (Path(output_dir) / version).exists():
        raise FileExistsError(
            f"La versión '{version}' ya existe en {output_dir}. Usa otro identificador.")

    df_crudo = leer_datos(ruta_datos, target)
    df = limpiar_entrenamiento(df_crudo, target)
    X, y = crear_features(df), df[target]

    X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size, seed)

    baseline = construir_baseline().fit(X_train, y_train)
    modelo = construir_modelo(n_estimators, seed).fit(X_train, y_train)
    metricas = {
        "baseline": evaluar(baseline, X_test, y_test),
        "modelo": evaluar(modelo, X_test, y_test),
    }

    metadata = armar_metadata(version, ruta_datos, len(df_crudo), len(df),
                              n_estimators, seed, test_size, len(X_train), len(X_test))
    rutas = guardar_version(modelo, metricas, metadata, output_dir, version)
    return rutas, metricas


def parsear_argumentos():
    p = argparse.ArgumentParser(
        description="Entrena el modelo y guarda una versión en <output-dir>/<version>/.")
    p.add_argument("--data", required=True, help="Ruta del CSV de datos etiquetados")
    p.add_argument("--target", default=TARGET, help="Variable objetivo (default: %(default)s)")
    p.add_argument("--output-dir", default="models", help="Carpeta de versiones (default: %(default)s)")
    p.add_argument("--version", required=True, help="Identificador de la versión, ej. v1")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: %(default)s)")
    p.add_argument("--test-size", type=float, default=0.2, help="Fracción para prueba (default: %(default)s)")
    p.add_argument("--n-estimators", type=int, default=300, help="Árboles del Random Forest (default: %(default)s)")
    return p.parse_args()


def main():
    args = parsear_argumentos()
    try:
        rutas, metricas = entrenar(args.data, args.target, args.output_dir, args.version,
                                   args.seed, args.test_size, args.n_estimators)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Versión '{args.version}' entrenada.")
    for nombre, m in metricas.items():
        print(f"  {nombre}: MAE={m['mae']:.0f}  RMSE={m['rmse']:.0f}  R2={m['r2']:.3f}")
    print("Archivos generados:")
    for ruta in rutas.values():
        print(f"  {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())