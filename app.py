import streamlit as st
import pickle

# ======================================
# CARGAR MODELO
# ======================================

with open("modelo_spam.pkl", "rb") as f:
    model = pickle.load(f)

# ======================================
# INTERFAZ STREAMLIT
# ======================================

st.set_page_config(
    page_title="Detector de Spam",
    page_icon="🚨"
)

st.title("🚨 Detector de Spam de YouTube")

st.write("Introduce un comentario y el modelo detectará si es spam o no.")

# Caja de texto
texto = st.text_area("Comentario")

# Botón
if st.button("Analizar"):

    if texto.strip() == "":
        st.warning("Por favor, escribe un comentario.")
    else:

        pred = model.predict([texto])[0]

        prob = model.predict_proba([texto])[0][1]

        st.subheader("Resultado")

        if int(pred) == 1:
            st.error("⚠️ El comentario es SPAM")
        else:
            st.success("✅ El comentario NO es spam")

        st.write(f"Probabilidad de spam: {round(float(prob) * 100, 2)}%")