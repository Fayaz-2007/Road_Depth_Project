# Road Surface Displacement Detection

A deep learning and regression-based system for detecting and measuring road surface displacement (potholes, bumps, etc.) from images.

## Project Structure

```
Road-Surface-Displacement/
│
├── dataset/
│   ├── bump/              # Bump images
│   ├── potholes/          # Pothole images
│   ├── road/              # Normal road images
│   └── labels.xlsx        # Image labels with displacement values
│
├── Python_CNN_Model/
│   ├── train_model.py           # CNN training script (TensorFlow/Keras)
│   ├── road_displacement_model.h5   # Trained CNN model
│   ├── sample.jpg               # Sample test image
│   ├── sample1.jpg              # Sample test image
│   └── requirements.txt        # Python dependencies
│
├── MATLAB_Regression_Model/
│   ├── train_regression.m       # MATLAB regression training script
│   ├── predict_image.m          # MATLAB prediction script
│   └── final_regression_model.mat   # Trained MATLAB model
│
├── .gitignore
└── README.md
```

## Setup

### Python CNN Model

```bash
cd Python_CNN_Model
pip install -r requirements.txt
python train_model.py
```

### MATLAB Regression Model

Open MATLAB, navigate to `MATLAB_Regression_Model/`, and run:

```matlab
train_regression
predict_image
```

## Dataset

The `dataset/` folder contains labeled road surface images organized by category:
- **bump/** — road bump images (250 images)
- **potholes/** — pothole images (915 images)
- **road/** — normal road images (250 images)

Labels with displacement measurements (in cm) are stored in `dataset/labels.xlsx`.

## Technologies

- **Python**: TensorFlow/Keras CNN for image classification & regression
- **MATLAB**: Boosted ensemble regression model for displacement prediction
