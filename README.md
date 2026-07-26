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
│   ├── live_predict.py          # Real-time camera prediction script
│   ├── road_displacement_model.h5   # Trained CNN model
│   ├── sample.jpg               # Sample test image
│   ├── sample1.jpg              # Sample test image
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile               # Docker configuration
│   └── .dockerignore
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

#### Run Live Camera Prediction
To test the model in real-time using your USB camera/webcam:
```bash
python live_predict.py
```
*(Press 'q' to quit the video window)*

### Running with Docker (Recommended for Beginners)

If you don't want to install Python and all the dependencies locally, you can easily run the Streamlit application using Docker! 

**Prerequisites:** You must have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your machine.

1. **Build the Docker Image:**
   Open your terminal, navigate into the Python model folder, and build the image:
   ```bash
   cd Python_CNN_Model
   docker build -t road-depth-predictor .
   ```

2. **Run the Container:**
   Once built, start the container and expose the web app port:
   ```bash
   docker run -p 8501:8501 road-depth-predictor
   ```

3. **Access the Web App:**
   Open your web browser and go to: `http://localhost:8501`

*(Note: The "Open Live Camera" button inside the web app requires a physical display. If you are running via Docker, that specific button will not work because Docker cannot natively access your host's USB camera and screen. You can still use the "Image Upload" feature perfectly!)*

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

- **Python**: TensorFlow/Keras CNN for image classification & regression, OpenCV for real-time camera feed
- **MATLAB**: Boosted ensemble regression model for displacement prediction
