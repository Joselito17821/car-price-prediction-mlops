"""Guardado y carga de versiones del modelo."""
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import joblib

from src.features import CAT_COLS, COL_FECHA, COLUMNAS_ENTRADA, TARGET


def esquema_entrada():
    """Columnas que debe traer el CSV de predicción y su tipo esperado."""
    esquema = {}
    for col in COLUMNAS_ENTRADA:
        if col == COL_FECHA:
            esquema[col] = "fecha"
        elif col in CAT_COLS:
            esquema[col] = "categorica"
        else:
            esquema[col] = "numerica"
    return esquema


def hash_commit():
    """Hash del commit actual; en Docker se puede pasar con la variable GIT_COMMIT."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        salida = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, timeout=5)
        return salida.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "desconocido"


def armar_metadata(version, ruta_datos, n_filas_crudas, n_filas_limpias,
                   n_estimators, seed, test_size, n_train, n_test):
    return {
        "version": version,
        "fecha_creacion": datetime.now().isoformat(timespec="seconds"),
        "tipo_problema": "regresion",
        "variable_objetivo": TARGET,
        "algoritmo": "RandomForestRegressor sobre log1p del objetivo",
        "parametros": {"n_estimators": n_estimators, "random_state": seed},
        "semilla": seed,
        "columnas_entrada": esquema_entrada(),
        "datos": {
            "fuente": "Kaggle: vrajesh0sharma7/used-car-price-prediction",
            "archivo": Path(ruta_datos).name,
            "observaciones_originales": n_filas_crudas,
            "observaciones_tras_limpieza": n_filas_limpias,
        },
        "separacion": {
            "estrategia": "train_test_split aleatorio",
            "test_size": test_size,
            "n_train": n_train,
            "n_test": n_test,
        },
        "metrica_evaluacion": ["mae", "rmse", "r2"],
        "python": platform.python_version(),
        "dependencias": "requirements.txt",
        "commit": hash_commit(),
    }


def _escribir_json(ruta, contenido):
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def guardar_version(modelo, metricas, metadata, output_dir, version):
    """Crea output_dir/version/ con modelo, métricas y metadatos.
    Si la versión ya existe, falla sin tocar nada."""
    carpeta = Path(output_dir) / version
    if carpeta.exists():
        raise FileExistsError(
            f"La versión '{version}' ya existe en {carpeta}. Usa otro identificador.")
    carpeta.mkdir(parents=True)
    rutas = {
        "modelo": carpeta / "modelo.joblib",
        "metrics": carpeta / "metrics.json",
        "metadata": carpeta / "metadata.json",
    }
    joblib.dump(modelo, rutas["modelo"], compress=3)
    _escribir_json(rutas["metrics"], metricas)
    _escribir_json(rutas["metadata"], metadata)
    return rutas


def cargar_version(ruta):
    """Carga (modelo, metadata) desde la carpeta de la versión o desde su modelo.joblib."""
    ruta = Path(ruta)
    if ruta.is_dir():
        ruta = ruta / "modelo.joblib"
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el modelo en {ruta}")
    ruta_meta = ruta.parent / "metadata.json"
    if not ruta_meta.is_file():
        raise FileNotFoundError(f"Falta metadata.json junto al modelo en {ruta.parent}")
    try:
        modelo = joblib.load(ruta)
    except Exception as e:
        raise ValueError(f"No se pudo leer el modelo {ruta}: {e}") from e
    metadata = json.loads(ruta_meta.read_text(encoding="utf-8"))
    return modelo, metadata