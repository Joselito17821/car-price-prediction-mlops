"""Construcción del pipeline, partición de datos y evaluación."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import CAT_COLS, NUM_COLS


def dividir_datos(X, y, test_size, seed):
    """Separa train/test antes de ajustar cualquier transformación."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def construir_preprocesador():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_COLS),
    ])


def construir_baseline():
    return Pipeline([
        ("prep", construir_preprocesador()),
        ("model", DummyRegressor(strategy="median")),
    ])


def construir_modelo(n_estimators, seed):
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("prep", construir_preprocesador()),
            ("model", RandomForestRegressor(
                n_estimators=n_estimators, random_state=seed, n_jobs=-1)),
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def evaluar(modelo, X, y):
    """Devuelve MAE, RMSE y R2 como floats (listos para guardar en JSON)."""
    pred = modelo.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }