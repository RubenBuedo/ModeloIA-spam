from streamlit as st
from pydantic import BaseModel
import pickle


# ======================================
# CARGAR MODELO
# ======================================

with open("modelo_spam.pkl", "rb") as f:
    model = pickle.load(f)


# ======================================
# INICIAR FASTAPI
# ======================================

app = FastAPI(
    title="YouTube Spam Detector",
    version="1.0"
)

from fastapi.middleware.cors import CORSMiddleware

# Añadir configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# MODELO DE ENTRADA
# ======================================

class Comentario(BaseModel):
    texto: str


# ======================================
# RUTA PRINCIPAL
# ======================================

@app.get("/")
def home():
    return {
        "mensaje": "API detector de spam funcionando"
    }


# ======================================
# PREDICCIÓN
# ======================================

@app.post("/predict")
def predict(data: Comentario):

    texto = data.texto

    pred = model.predict([texto])[0]

    prob = model.predict_proba([texto])[0][1]

    return {
        "texto": texto,
        "spam": int(pred),
        "probabilidad_spam": round(float(prob), 4)
    }
