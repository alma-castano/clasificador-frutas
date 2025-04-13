import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import os
import zipfile
import gdown

st.title("Clasificador de Frutas 🍓🍍🍌")

# Rutas del archivo
modelo_zip = "modelo_frutasFINAL.zip"
modelo_path = "modelo_frutasFINAL.keras"

# Descargar y descomprimir el modelo si no existe
if not os.path.exists(modelo_path):
    with st.spinner("Descargando y descomprimiendo el modelo..."):
        file_id = "1TcsxaaQur7Fc2lJZKu5OAiuFv_ytdcrn"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, modelo_zip, quiet=False, fuzzy=True)
        with zipfile.ZipFile(modelo_zip, 'r') as zip_ref:
            zip_ref.extractall()
        st.success("✅ Modelo descargado y listo")

# Cargar el modelo solo una vez
@st.cache_resource
def cargar_modelo():
    return load_model(modelo_path)

modelo = cargar_modelo()

# Lista de clases
class_names = [
    'Tomato', 'Grape', 'Onion', 'Plum', 'Cantaloupe', 'Kiwi', 'Avocado', 'Mango',
    'Pomegranate', 'Pepper', 'Blueberry', 'Clementine', 'Peach', 'Cherry', 'Cucumber',
    'Apricot', 'Apple', 'Papaya', 'Potato', 'Pineapple', 'Lemon', 'Limes', 'Passion',
    'Banana', 'Pear', 'Watermelon', 'Raspberry', 'Orange', 'Cactus', 'Strawberry',
    'Corn'
]

# Subida de imagen
uploaded_file = st.file_uploader("📷 Subí una imagen de fruta", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen subida", use_container_width=True)

    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = modelo.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    st.markdown(f"### 🧠 Predicción: **{predicted_class}**")
    st.markdown(f"📊 Confianza: **{confidence * 100:.2f}%**")

