import streamlit as st
import numpy as np
from PIL import Image
from predict import load_model, predict_displacement

st.set_page_config(page_title="Road Displacement Predictor")

st.title("🚗 Road Displacement Prediction System")

# Load model once
model = load_model()

uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save temp image
    temp_path = "temp.jpg"
    img.save(temp_path)

    displacement = predict_displacement(model, temp_path)

    if displacement is not None:
        st.success(f"Predicted Displacement: {displacement:.2f} cm")