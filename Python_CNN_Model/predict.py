import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Settings
IMAGE_SIZE = 128
MODEL_PATH = "road_displacement_model.h5"


def load_model():
    """Load the trained CNN model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Run train_model.py first to train and save the model.")
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def predict_displacement(model, image_path):
    """Predict road surface displacement for a single image."""
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return None

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array, verbose=0)
    return prediction[0][0]


def main():
    # Accept image path as command-line argument or use default samples
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default: try sample images in this folder
        for sample in ["sample.jpg", "sample1.jpg"]:
            if os.path.exists(sample):
                image_path = sample
                break
        else:
            print("Usage: python predict.py <image_path>")
            print("  or place sample.jpg / sample1.jpg in this folder.")
            sys.exit(1)

    print("Loading model...")
    model = load_model()

    print(f"Predicting displacement for: {image_path}")
    displacement = predict_displacement(model, image_path)

    if displacement is not None:
        print(f"Predicted Displacement: {displacement:.2f} cm")


if __name__ == "__main__":
    main()
