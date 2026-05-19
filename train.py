# ======================================
# IMPORTAR LIBRERÍAS
# ======================================

import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ======================================
# CARGAR DATASET
# ======================================

file_path = "BD_limpia_final.xlsx"

df = pd.read_excel(file_path)

print("\nPrimeras filas:")
print(df.head())

print("\nColumnas:")
print(df.columns)


# ======================================
# RENOMBRAR COLUMNAS
# ======================================

df = df.rename(columns={
    "CONTENT": "comentario",
    "CLASS": "spam"
})


# ======================================
# VALIDAR COLUMNAS
# ======================================

required_columns = ["comentario", "spam"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Falta la columna: {col}")


# ======================================
# LIMPIEZA DE DATOS
# ======================================

df = df.dropna(subset=["comentario", "spam"])

df["comentario"] = df["comentario"].astype(str)


# ======================================
# LIMPIEZA DE TEXTO
# ======================================

def limpiar_texto(texto):

    texto = texto.lower()

    # eliminar urls
    texto = re.sub(r"http\\S+", " ", texto)

    # eliminar menciones
    texto = re.sub(r"@\\w+", " ", texto)

    # eliminar caracteres raros
    texto = re.sub(r"[^a-zA-ZáéíóúñüÁÉÍÓÚÑ0-9 ]", " ", texto)

    # eliminar espacios extra
    texto = re.sub(r"\\s+", " ", texto).strip()

    return texto


df["comentario"] = df["comentario"].apply(limpiar_texto)


# ======================================
# DISTRIBUCIÓN DE CLASES
# ======================================

print("\nDistribución de clases:")
print(df["spam"].value_counts())


# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    df["comentario"],
    df["spam"],
    test_size=0.2,
    random_state=42,
    stratify=df["spam"]
)


# ======================================
# PIPELINE NLP
# ======================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ======================================
# ENTRENAMIENTO
# ======================================

print("\nEntrenando modelo...")

pipeline.fit(X_train, y_train)

print("Modelo entrenado correctamente")


# ======================================
# PREDICCIONES
# ======================================

y_pred = pipeline.predict(X_test)


# ======================================
# MÉTRICAS
# ======================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nF1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


# ======================================
# MATRIZ DE CONFUSIÓN
# ======================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión")

plt.show()


# ======================================
# ANÁLISIS DE ERRORES
# ======================================

df_test = pd.DataFrame({
    "comentario": X_test,
    "real": y_test,
    "pred": y_pred
})

# Falsos positivos
fp = df_test[
    (df_test["real"] == 0) &
    (df_test["pred"] == 1)
]

# Falsos negativos
fn = df_test[
    (df_test["real"] == 1) &
    (df_test["pred"] == 0)
]

print("\n===== FALSOS POSITIVOS =====")
print(fp.head(10))

print("\n===== FALSOS NEGATIVOS =====")
print(fn.head(10))


# ======================================
# PRUEBAS MANUALES
# ======================================

comentarios = [
    "Earn money fast by clicking here!!!",
    "Very good video thanks",
    "Check out my channel and subscribe",
    "Great content bro"
]

for texto in comentarios:

    pred = pipeline.predict([texto])[0]

    prob = pipeline.predict_proba([texto])[0][1]

    print("\n========================")
    print("Comentario:", texto)
    print("Spam:", pred)
    print("Probabilidad spam:", round(prob, 4))


# ======================================
# GUARDAR MODELO
# ======================================

with open("modelo_spam.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModelo guardado correctamente")
