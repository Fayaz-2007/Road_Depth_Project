import os
import sys
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Settings
IMAGE_SIZE = 128
MODEL_PATH = "road_displacement_model.h5"
WINDOW_SIZE = 5  # For smoothing

def load_existing_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Ensure you are using the existing trained model.")
        sys.exit(1)
        
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def predict_video_sequence(video_path, model):
    if not os.path.exists(video_path):
        print(f"Error: Video '{video_path}' not found.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    print(f"Processing video: {video_path}")
    
    records = []
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        timestamp = frame_id / fps
        
        # Preprocess for CNN
        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(color_converted)
        resized_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(resized_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict displacement directly in cm (no target normalization was used in original training)
        pred_cm = model.predict(img_array, verbose=0)[0][0]
        
        records.append({
            'video_id': os.path.basename(video_path),
            'frame_id': frame_id,
            'timestamp': round(timestamp, 3),
            'predicted_displacement_cm': pred_cm
        })
        
        if frame_id % 10 == 0:
            print(f"  Processed {frame_id} frames...")
            
    cap.release()
    
    if not records:
        print("Error: No frames could be read from the video.")
        return None
        
    df = pd.DataFrame(records)
    
    # Normalization / Temporal Processing: Apply moving average to smooth predictions
    df['smoothed_displacement_cm'] = df['predicted_displacement_cm'].rolling(window=WINDOW_SIZE, min_periods=1, center=True).mean()
    
    return df

def process_results(df, video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Aggregation: Extract maximum from the smoothed sequence
    # As the camera gets closer, the anomaly appears larger, peaking before leaving the frame.
    final_estimate = df['smoothed_displacement_cm'].max()
    
    # Save CSV
    csv_path = f"{video_name}_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved prediction sequence to {csv_path}")
    
    # Generate Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['frame_id'], df['predicted_displacement_cm'], label='Raw Prediction', alpha=0.5, color='gray')
    plt.plot(df['frame_id'], df['smoothed_displacement_cm'], label=f'Smoothed (Window={WINDOW_SIZE})', color='blue', linewidth=2)
    plt.axhline(final_estimate, color='red', linestyle='--', label=f'Final Estimate ({final_estimate:.2f} cm)')
    
    plt.title(f"Displacement Prediction Sequence for {video_name}")
    plt.xlabel("Frame Number")
    plt.ylabel("Predicted Displacement (cm)")
    plt.legend()
    plt.grid(True)
    
    plot_path = f"{video_name}_prediction_curve.png"
    plt.savefig(plot_path)
    print(f"Saved prediction plot to {plot_path}")
    
    print("\n--- Final Video-Level Measurement ---")
    print(f"Derived Estimated Displacement: {final_estimate:.2f} cm")
    
    return final_estimate

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_video.py <path_to_video.mp4>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    print("Loading existing CNN model...")
    model = load_existing_model()
    
    df = predict_video_sequence(video_path, model)
    
    if df is not None:
        process_results(df, video_path)

if __name__ == "__main__":
    main()
