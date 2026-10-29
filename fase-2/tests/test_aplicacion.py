"""Pruebas básicas de la aplicación: entrenamiento, predicción, nulos, errores y versiones.

Se ejecutan desde fase-2/ con:  python -m pytest
Usan datos sintéticos pequeños y directorios temporales (no tocan models/ ni data/).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import predict
from src.features import TARGET, crear_features
from src.modelo import construir_modelo
from src.versiones import armar_metadata, cargar_version, guardar_version
from train import entrenar

RAIZ = Path(__file__).resolve().parent.parent  # carpeta fase-2/


def crear_datos(n=300, seed=0):
    """Dataset sintético con el mismo esquema de entrada del proyecto, más sale_price."""
    rng = np.random.default_rng(seed)
    yr = rng.integers(2008, 2021, n)
    fecha = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    df = pd.DataFrame({
        "yr_mfr": yr,
        "kms_run": rng.integers(1000, 150000, n),
        "total_owners": rng.integers(1, 5, n),
        "warranty_avail": rng.choice([True, False], n),
        "ad_created_on": fecha.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3],
        "fuel_type": rng.choice(["petrol", "diesel", "cng"], n),
        "city": rng.choice(["noida", "delhi", "mumbai"], n),
        "body_type": rng.choice(["hatchback", "sedan", "suv"], n),
        "transmission": rng.choice(["manual", "automatic"], n),
        "source": rng.choice(["inperson_sale", "online"], n),
        "car_rating": rng.choice(["great", "good", "fair"], n),
        "fitness_certificate": rng.choice([True, False], n),
        "make": rng.choice(["maruti", "hyundai", "honda"], n),
        "model": rng.choice(["swift", "i20", "city"], n),
        "registered_state": rng.choice(["delhi", "maharashtra"], n),
    })
    df[TARGET] = (300000 + (yr - 2008) * 20000 + rng.normal(0, 30000, n)).astype(int)
    df.loc[rng.choice(n, 10, replace=False), "transmission"] = None  # nulos reales
    return df


def hash_archivo(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    """Entrena una versión v1 pequeña una sola vez y la comparte entre pruebas."""
    base = tmp_path_factory.mktemp("app")
    datos = base / "datos.csv"
    crear_datos().to_csv(datos, index=False)
    models = base / "models"
    rutas, metricas = entrenar(str(datos), TARGET, str(models), "v1", 42, 0.2, 5)
    return {"base": base, "datos": datos, "models": models,
            "rutas": rutas, "metricas": metricas}


def test_entrenamiento_y_almacenamiento(entorno):
    rutas = entorno["rutas"]
    for ruta in rutas.values():
        assert Path(ruta).is_file()

    metricas = json.loads(Path(rutas["metrics"]).read_text(encoding="utf-8"))
    assert set(metricas) == {"baseline", "modelo"}
    for m in metricas.values():
        assert set(m) == {"mae", "rmse", "r2"}
    assert metricas["modelo"]["mae"] < metricas["baseline"]["mae"]

    metadata = json.loads(Path(rutas["metadata"]).read_text(encoding="utf-8"))
    assert metadata["version"] == "v1"
    assert metadata["semilla"] == 42
    assert "columnas_entrada" in metadata

    modelo, _ = cargar_version(rutas["modelo"])
    assert hasattr(modelo, "predict")


def test_carga_y_prediccion(tmp_path):
    df = crear_datos()
    modelo = construir_modelo(5, 42).fit(crear_features(df), df[TARGET])
    metadata = armar_metadata("t1", "datos.csv", len(df), len(df), 5, 42, 0.2, len(df), 0)
    guardar_version(modelo, {}, metadata, tmp_path / "models", "t1")

    cargado, _ = cargar_version(tmp_path / "models" / "t1")
    nuevos = df.drop(columns=[TARGET]).head(25)
    esperado = modelo.predict(crear_features(nuevos))
    obtenido = predict.predecir(cargado, nuevos)

    assert len(obtenido) == len(nuevos)  # una predicción por fila
    np.testing.assert_allclose(obtenido, esperado)


def test_valores_faltantes(entorno):
    modelo, metadata = cargar_version(entorno["models"] / "v1")
    nuevos = crear_datos(20, seed=1).drop(columns=[TARGET])
    nuevos.loc[0, "kms_run"] = np.nan
    nuevos.loc[1, "ad_created_on"] = np.nan
    nuevos.loc[2, "transmission"] = None

    predict.validar_entrada(nuevos, metadata["columnas_entrada"])  # un nulo no es error
    pred = predict.predecir(modelo, nuevos)

    assert len(pred) == 20  # no se eliminó ninguna fila
    assert not np.isnan(pred).any()


def test_entrada_invalida(entorno, tmp_path):
    nuevos = crear_datos(10, seed=2).drop(columns=[TARGET, "make"])
    _, metadata = cargar_version(entorno["models"] / "v1")
    with pytest.raises(ValueError, match="Faltan columnas"):
        predict.validar_entrada(nuevos, metadata["columnas_entrada"])

    entrada = tmp_path / "sin_make.csv"
    salida = tmp_path / "salida.csv"
    nuevos.to_csv(entrada, index=False)
    resultado = subprocess.run(
        [sys.executable, "predict.py", "--model", str(entorno["models"] / "v1"),
         "--input", str(entrada), "--output", str(salida)],
        cwd=RAIZ, capture_output=True, text=True)

    assert resultado.returncode != 0
    assert "Faltan columnas" in resultado.stderr
    assert not salida.exists()  # no deja resultados que parezcan válidos


def test_versionamiento(tmp_path):
    datos = tmp_path / "datos.csv"
    crear_datos().to_csv(datos, index=False)
    models = tmp_path / "models"

    entrenar(str(datos), TARGET, str(models), "v1", 42, 0.2, 5)
    hash_v1 = hash_archivo(models / "v1" / "modelo.joblib")
    entrenar(str(datos), TARGET, str(models), "v2", 7, 0.2, 5)

    assert (models / "v1").is_dir() and (models / "v2").is_dir()
    cargar_version(models / "v1")
    cargar_version(models / "v2")
    assert hash_archivo(models / "v1" / "modelo.joblib") == hash_v1  # v1 intacta

    with pytest.raises(FileExistsError):
        entrenar(str(datos), TARGET, str(models), "v1", 1, 0.2, 5)
    modelo, _ = cargar_version(models / "v1")
    with pytest.raises(FileExistsError):  # guardar_version también rechaza el id repetido
        guardar_version(modelo, {}, {}, models, "v1")
    assert hash_archivo(models / "v1" / "modelo.joblib") == hash_v1  # no se sobrescribió