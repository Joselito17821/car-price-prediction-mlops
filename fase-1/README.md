# Fase 1 — Modelo predictivo: Precio de autos usados

## Integrantes
- Jose Manuel Londoño Castaño
- Juan Jose Gomez Castaño

## Descripción del problema
Predecir el precio de venta de un auto usado a partir de sus características (marca, modelo, año, kilometraje, ciudad, etc.).

## Fuente del dataset
[Used Car Price Prediction — Kaggle](https://www.kaggle.com/datasets/vrajesh0sharma7/used-car-price-prediction)

## Objetivo del modelo
Estimar `sale_price` (precio en rupias indias) para autos no vistos por el modelo.

## Algoritmo utilizado
Random Forest Regressor, entrenado sobre `log(sale_price)`, comparado contra un baseline (Dummy Regressor).

## Métrica empleada
MAE, RMSE y R², calculadas sobre un conjunto de prueba separado (20% de los datos).

## Principales resultados
| Métrica | Baseline | Random Forest |
|---|---|---|
| MAE | 179.222 | 42.956 |
| RMSE | 278.571 | 71.773 |
| R² | -0,055 | 0,930 |

## Instrucciones para ejecutar

1. Crear y activar un entorno virtual:
```bash
python -m venv .venv
```
En Windows:
```bash
.venv\Scripts\activate
```
En Mac/Linux:
```bash
source .venv/bin/activate
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```
