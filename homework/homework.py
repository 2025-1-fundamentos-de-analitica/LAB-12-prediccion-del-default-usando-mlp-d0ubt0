# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    make_scorer,
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


def load_and_clean(path):
    df = pd.read_csv(path, compression="zip")
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()

    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]

    X = df.drop(columns=["default"])
    y = df["default"]

    return X, y


def save_model_gzip(model, path="files/models/model.pkl.gz"):
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def save_classification_metrics(
    model, x_train, y_train, x_test, y_test, path="files/output/metrics.json"
):
    metrics = []
    model = model.best_estimator_

    for X, y, name in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
        y_pred = model.predict(X)
        entry = {
            "type": "metrics",
            "dataset": name,
            "precision": precision_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
        }
        metrics.append(entry)

    with open(path, "w") as f:
        for entry in metrics:
            json.dump(entry, f)
            f.write("\n")


def create_dirs():
    if not os.path.exists("files/models"):
        os.mkdir("files/models")

    if not os.path.exists("files/output"):
        os.mkdir("files/output")


def save_confusion_matrices(
    model, x_train, y_train, x_test, y_test, path="files/output/metrics.json"
):
    results = []
    model = model.best_estimator_

    for X, y, name in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
        y_pred = model.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        result = {
            "type": "cm_matrix",
            "dataset": name,
            "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
            "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
        }
        results.append(result)

    with open(path, "a") as f:
        for entry in results:
            json.dump(entry, f)
            f.write("\n")


x_train, y_train = load_and_clean("files/input/train_data.csv.zip")
x_test, y_test = load_and_clean("files/input/test_data.csv.zip")

categorical_features = (
    x_train.select_dtypes(include=["object", "category", "int64"])
    .columns[x_train.nunique() < 10]
    .tolist()
)

numerical_features = [col for col in x_train.columns if col not in categorical_features]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ],
    remainder="passthrough",
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(f_classif)),
        ("pca", PCA()),
        ("classifier", MLPClassifier(max_iter=20000, random_state=42)),
    ]
)


param_grid = {
    "pca__n_components": [None],
    "selector__k": [20],
    "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
    "classifier__alpha": [0.25, 0.26, 0.27, 0.28],
    "classifier__learning_rate_init": [0.001],
}

scorer = make_scorer(balanced_accuracy_score)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring=scorer,
    n_jobs=-1,
    verbose=4,
    refit=True,
)

grid_search.fit(x_train, y_train)


create_dirs()

print(grid_search.best_params_)
save_model_gzip(grid_search)

save_classification_metrics(grid_search, x_train, y_train, x_test, y_test)
save_confusion_matrices(grid_search, x_train, y_train, x_test, y_test)
