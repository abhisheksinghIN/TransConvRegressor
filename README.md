## 🌱 TransConvRegressor: LAI Estimation with Baackscatter Signals

This repository implements **TransConvRegressor**, a 1D TransUNet-based regressor for **Leaf Area Index (LAI)** estimation using temporal Sentinel-1 backscatter signals and ancillary features. It includes training, evaluation, and prediction pipelines.

---

## 📂 Repository Structure
- **network.py**: Contains the `model` function and helper functions for building the 1D TransConvRegressor model.
- **train.py**: Loads data, applies preprocessing, creates train/val/test splits, trains the model, saves the model, scaler and evaluation metrics.
- **prediction.py**: Loads the trained model, applies preprocessing, makes predictions on new data and saves outputs.
- **requirements.txt**: Lists all Python packages required to run the scripts.

## Installation
git clone https://github.com/abhisheksinghIN/TransConvRegressor.git
cd TransConvRegressor
pip install -r requirements.txt

## Training, Validation and Test
python train.py
- This will initiate the training process, including data preprocessing, model training, and validation.

## Prediction
python prediction.py

## Model Architecture
The core model is a 1D Transformer-based architecture, combining convolutional layers with transformer-based attention mechanisms to capture spatial and temporal dependencies in the Sentinel-1 backscatter signals.

