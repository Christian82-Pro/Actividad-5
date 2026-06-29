import os
import shutil
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datos_prep import cargar_datos, limpiar_datos

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

os.makedirs("datos/datos_ini", exist_ok=True)
os.makedirs("datos/datos_limp", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

if os.path.exists("mlruns"):
    shutil.rmtree("mlruns")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("clasificacion_tickets_soporte")

tabla = cargar_datos()
tabla.head(1000).to_csv("datos/datos_ini/muestra_tickets.csv", index=False)

tabla = limpiar_datos(tabla)
tabla.to_csv("datos/datos_limp/tickets_limpios.csv", index=False)

tabla = tabla.sample(n=min(3000, len(tabla)), random_state=42)

x = tabla["texto"]
y = tabla["queue"]

plt.figure(figsize=(10, 5))
tabla["queue"].value_counts().plot(kind="bar")
plt.title("Distribucion de tickets por categoria")
plt.xlabel("Categoria")
plt.ylabel("Cantidad")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("artifacts/distribucion_categorias.png")
plt.close()

tabla["longitud_texto"] = tabla["texto"].str.len()

plt.figure(figsize=(8, 5))
plt.hist(tabla["longitud_texto"], bins=30)
plt.title("Longitud de los tickets")
plt.xlabel("Longitud")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig("artifacts/longitud_tickets.png")
plt.close()

x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

validacion = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

def entrenar_y_guardar(nombre_modelo, proceso, parametros):
    busqueda = GridSearchCV(
        estimator=proceso,
        param_grid=parametros,
        cv=validacion,
        scoring="f1_macro",
        n_jobs=-1
    )

    with mlflow.start_run(run_name=nombre_modelo):
        busqueda.fit(x_entrenamiento, y_entrenamiento)

        mejor_modelo = busqueda.best_estimator_
        predicciones = mejor_modelo.predict(x_prueba)

        accuracy = accuracy_score(y_prueba, predicciones)
        precision = precision_score(y_prueba, predicciones, average="macro", zero_division=0)
        recall = recall_score(y_prueba, predicciones, average="macro", zero_division=0)
        f1 = f1_score(y_prueba, predicciones, average="macro", zero_division=0)

        try:
            probas = mejor_modelo.predict_proba(x_prueba)
            roc_auc = roc_auc_score(y_prueba, probas, multi_class="ovr", average="macro")
        except:
            roc_auc = 0

        reporte = classification_report(y_prueba, predicciones, zero_division=0, output_dict=True)
        reporte_tabla = pd.DataFrame(reporte).transpose()

        ruta_reporte = "artifacts/reporte_" + nombre_modelo + ".csv"
        reporte_tabla.to_csv(ruta_reporte)

        matriz = confusion_matrix(y_prueba, predicciones)

        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz, cmap="Blues")
        plt.title("Matriz de confusion - " + nombre_modelo)
        plt.xlabel("Prediccion")
        plt.ylabel("Real")
        plt.tight_layout()

        ruta_matriz = "artifacts/matriz_" + nombre_modelo + ".png"
        plt.savefig(ruta_matriz)
        plt.close()

        mlflow.log_param("modelo", nombre_modelo)
        mlflow.log_params(busqueda.best_params_)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("roc_auc_macro", roc_auc)

        mlflow.log_artifact(ruta_reporte)
        mlflow.log_artifact(ruta_matriz)
        mlflow.log_artifact("artifacts/distribucion_categorias.png")
        mlflow.log_artifact("artifacts/longitud_tickets.png")

        mlflow.sklearn.log_model(mejor_modelo, "modelo")

        return {
            "modelo": nombre_modelo,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "mejores_parametros": str(busqueda.best_params_)
        }

proceso_logistica = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("modelo", LogisticRegression(max_iter=1000))
])

parametros_logistica = {
    "tfidf__max_features": [3000, 5000],
    "modelo__C": [0.1, 1, 10]
}

proceso_bosque = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("modelo", RandomForestClassifier(random_state=42))
])

parametros_bosque = {
    "tfidf__max_features": [3000, 5000],
    "modelo__n_estimators": [100, 200],
    "modelo__max_depth": [None, 20]
}

resultados = []

resultados.append(entrenar_y_guardar("Regresion_Logistica", proceso_logistica, parametros_logistica))
resultados.append(entrenar_y_guardar("Random_Forest", proceso_bosque, parametros_bosque))

tabla_resultados = pd.DataFrame(resultados)
tabla_resultados.to_csv("results/resultados_mlflow.csv", index=False)

tabla_resultados.set_index("modelo")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar")
plt.title("Comparacion de modelos")
plt.ylabel("Valor")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/comparacion_modelos.png")
plt.close()

print(tabla_resultados)
print("Proceso terminado")
