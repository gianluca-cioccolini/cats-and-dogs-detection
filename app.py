import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import tempfile

st.title("Object Detection con YOLOv8 🐱🐶")

# Carica modello
model = YOLO("best.pt")

# Carica immagine
uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        results = model.predict(source=tmp_file.name)

    # Mostra immagine originale
    st.image(img_array, caption='Immagine caricata', use_column_width=True)

    # Ottieni immagine con bounding box
    result_bgr = results[0].plot()         # <-- restituisce BGR
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)  # <-- converti in RGB

    st.image(result_rgb, caption='Rilevamento oggetti', use_column_width=True)

