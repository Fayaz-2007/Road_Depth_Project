import cv2
import numpy as np
import streamlit as st
from PIL import Image
from predict import load_model

# Settings
IMAGE_SIZE = 128

def run_live_prediction(model=None, camera_index=0):
    if model is None:
        print("Loading model...")
        model = load_model()

    print(f"Opening webcam at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        error_msg = "Error: Could not open camera. Please check if your USB camera is connected or if another app is using it."
        print(error_msg)
        st.error(error_msg)
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert frame from OpenCV BGR format to PIL Image format (RGB)
        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(color_converted)
        
        # Preprocess exactly as in predict.py
        resized_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(resized_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict displacement
        prediction = model.predict(img_array, verbose=0)
        displacement = prediction[0][0]

        # Overlay the text on the frame
        text = f"Displacement: {displacement:.2f} cm"
        cv2.putText(
            frame, 
            text, 
            (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), # Green color
            2, 
            cv2.LINE_AA
        )

        # Display the resulting frame
        cv2.imshow('Live Road Displacement Prediction', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_live_prediction()
