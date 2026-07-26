import streamlit as st
import numpy as np
from PIL import Image
from predict import load_model, predict_displacement

st.set_page_config(page_title="Road Displacement Predictor")

st.title("🚗 Road Displacement Prediction System")

# Load model once
model = load_model()

st.markdown("---")
st.subheader("Real-Time Prediction")
camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1, help="0 is usually the built-in webcam. Change to 1 or 2 for a USB camera.")

if st.button("Open Live Camera (Local Only)"):
    from live_predict import run_live_prediction
    st.info("Opening OpenCV camera window... Press 'q' in the window to quit.")
    run_live_prediction(model, camera_index)

st.markdown("---")
st.subheader("Image Upload Prediction")
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