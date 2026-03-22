import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image

# Settings
IMAGE_SIZE = 128
dataset_folder = os.path.join("..", "dataset")
labels_path = os.path.join(dataset_folder, "labels.xlsx")

# Category subfolders inside dataset/
category_dirs = {
    "bump": os.path.join(dataset_folder, "bump"),
    "potholes": os.path.join(dataset_folder, "potholes"),
    "road": os.path.join(dataset_folder, "road"),
}

# Load labels
df = pd.read_excel(labels_path)

# Prepare image data and labels
X = []
y = []

for index, row in df.iterrows():
    img_name = row["image_name"]

    # Search for the image across all category folders
    img_path = None
    for cat_dir in category_dirs.values():
        candidate = os.path.join(cat_dir, img_name)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"Skipping {img_name} (not found)")
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img) / 255.0  # Normalize
        X.append(img_array)
        y.append(row["displacement_cm"])
    except:
        print(f"Skipping {img_name}")

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # Regression output
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# Save model
model.save("road_displacement_model.h5")

print("Model training complete and saved.")
