## 🌱 TransConvRegressor: LAI Estimation with Baackscatter Signals

This repository implements **TransConvRegressor**, a 1D TransUNet-based regressor for **Leaf Area Index (LAI)** estimation using temporal Sentinel-1 backscatter signals and ancillary features. It includes training, evaluation, and prediction pipelines.

- Model Architecture
The core model is a 1D Transformer-based architecture, combining convolutional layers with transformer-based attention mechanisms to capture spatial and temporal dependencies in the Sentinel-1 backscatter signals.
<img width="3919" height="1470" alt="image" src="https://github.com/user-attachments/assets/f9de8c09-1b3e-4b2a-a651-179af88a4919" />



## 📂 Repository Structure
- **network.py**: Contains the `model` function and helper functions for building the 1D TransConvRegressor model.
- **train.py**: Loads data, applies preprocessing, creates train/val/test splits, trains the model, saves the model, scaler and evaluation metrics.
- **prediction.py**: Loads the trained model, applies preprocessing, makes predictions on new data and saves outputs.
- **requirements.txt**: Lists all Python packages required to run the scripts.

## Data Description
- S1 SAR (RTC): We downloaded the S1 RTC from the Microsoft Planetary Computer as netcdfs. RTC is radiometrically terrain corrected GRD data, to account for radiometrics effects resulting from the topography.
- Sentinel-2 derived LAI: We computed LAI from the S2 L2A files over selected years and tiles using the SNAP Biophysical Processor.
- Soil Moisture: We considered Surface Soil Moisture based on a combination of SAR and optical imagery [1]. We simulated the required soil moisture data over the AOI.
- Topographical Classes: Type of meadows are selected from LAFIS data. We have only considered colline, submontane, montane and subalpine for this study because most of managed grasslands is in this elevation range.
- AOI: LAFIS Grasslands over South Tyrol (Italy)
[1] F. Greifeneder, C. Notarnicola, and W. Wagner, “A machine learning-based approach for surface soil moisture estimations with google earth engine,” Remote Sensing, vol. 13, no. 11, p. 2099, 2021.

## Installation
- git clone https://github.com/abhisheksinghIN/TransConvRegressor.git
- cd TransConvRegressor
- pip install -r requirements.txt

## Training, Validation and Test
python train.py
- This will initiate the training process, including data preprocessing, model training, and validation.

## Prediction
python prediction.py
- This script loads the trained model, performs inference on the test data, and saves the predicted LAI values along with additional information to CSV files.
<img width="3876" height="2071" alt="image" src="https://github.com/user-attachments/assets/9815cae6-0b79-4aee-a3f1-29df30c1ada3" />


