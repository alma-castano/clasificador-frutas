import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

st.title("Clasificador de Frutas 🍓🍍🍌")

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return load_model("modelo_frutasV2.keras")

modelo = cargar_modelo()

# Lista de clases
class_names = [
    'Tomato', 'Grape', 'Onion', 'Plum', 'Cantaloupe', 'Kiwi', 'Avocado', 'Mango',
    'Pomegranate', 'Pepper', 'Blueberry', 'Clementine', 'Peach', 'Cherry', 'Cucumber',
    'Apricot', 'Apple', 'Papaya', 'Potato', 'Pineapple', 'Lemon', 'Limes', 'Passion',
    'Banana', 'Pear', 'Watermelon', 'Raspberry', 'Orange', 'Cactus', 'Strawberry',
    'Corn'
]

# Subir una imagen
uploaded_file = st.file_uploader("Subí una imagen de fruta", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)

    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = modelo.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    st.markdown(f"### 🧠 Predicción: **{predicted_class}**")
    st.markdown(f"Confianza: **{confidence * 100:.2f}%**")
