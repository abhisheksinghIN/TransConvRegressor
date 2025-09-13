import os
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MultiHeadAttention, LayerNormalization, Add, Input, GlobalAveragePooling1D, Conv1DTranspose, Concatenate, Dropout, Layer
from tensorflow.keras.layers import Cropping1D, ZeroPadding1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import gelu, relu
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from joblib import dump
from tensorflow.keras import backend as K
from time import time

# Define paths
path = "./workspace/"
filename_database = path + "fixed_classsid.csv"
path_exp = "./workspace/git/"

path_val = "./workspace/pred/"

#os.makedirs(path_val, exist_ok=True)

# Load the data
df_ = pd.read_csv(filename_database, sep=',')

#**************************************************************************************************
#*************************************** Preprocessing ********************************************
#**************************************************************************************************

df_['doy_sin'] = np.sin(2 * np.pi * df_['doy'] / 365)
df_['doy_cos'] = np.cos(2 * np.pi * df_['doy'] / 365)
df_['classid_sin'] = np.sin(2 * np.pi * df_['class_id'] / 8)
df_['classid_cos'] = np.cos(2 * np.pi * df_['class_id'] / 8)


df_["RTC_dsc_VH"] = 10 ** (df_["RTC_dsc_VH"] * 0.1)
df_["RTC_dsc_VV"] = 10 ** (df_["RTC_dsc_VV"] * 0.1)
df_["RTC_asc_VH"] = 10 ** (df_["RTC_asc_VH"] * 0.1)
df_["RTC_asc_VV"] = 10 ** (df_["RTC_asc_VV"] * 0.1)


df_["RTC_dsc_RVI"] = 4 * df_["RTC_dsc_VH"] / (df_["RTC_dsc_VV"] + df_["RTC_dsc_VH"])
df_["RTC_dsc_ratio"] = df_["RTC_dsc_VH"] / df_["RTC_dsc_VV"]

df_["RTC_asc_RVI"] = 4 * df_["RTC_asc_VH"] / (df_["RTC_asc_VV"] + df_["RTC_asc_VH"])
df_["RTC_asc_ratio"] = df_["RTC_asc_VH"] / df_["RTC_asc_VV"]

df_["RTC_dsc_prod"] = df_["RTC_dsc_VV"] * df_["RTC_dsc_VH"]
df_["RTC_asc_prod"] = df_["RTC_asc_VV"] * df_["RTC_asc_VH"]

df_["RTC_dsc_sum"] = df_["RTC_dsc_VV"] + df_["RTC_dsc_VH"]
df_["RTC_asc_sum"] = df_["RTC_asc_VV"] + df_["RTC_asc_VH"]


columns_to_scale = ["RTC_dsc_VV", "RTC_asc_VV", "RTC_dsc_VH", "RTC_asc_VH", 'RTC_dsc_RVI','RTC_asc_RVI', 'RTC_dsc_ratio', 'RTC_asc_ratio', 'RTC_dsc_prod', 'RTC_asc_prod', 'RTC_dsc_sum', 'RTC_asc_sum', 'SM', 'doy_sin', 'doy_cos', 'classid_sin', 'classid_cos']
scaler = StandardScaler()
df_[columns_to_scale] = scaler.fit_transform(df_[columns_to_scale])
dump(scaler, os.path.join(path_exp, 'scaler.pkl'))
print("Scaler saved as 'scaler.pkl'")

X_test = df_[["RTC_dsc_VV", "RTC_asc_VV", "RTC_dsc_VH", "RTC_asc_VH", 'RTC_dsc_RVI','RTC_asc_RVI', 'RTC_dsc_ratio', 'RTC_asc_ratio', 'RTC_dsc_prod', 'RTC_asc_prod', 'RTC_dsc_sum', 'RTC_asc_sum', 'SM', 'doy_sin', 'doy_cos', 'classid_sin', 'classid_cos']].values 
y_test = df_['LAI'].values 
    
#**************************************************************************************************
#************************** Prediction using Regressor Model (SATUNET) ****************************
#**************************************************************************************************

X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

def bounded_relu(x):
    return 8 * K.relu(x) / (1 + K.relu(x))

def custom_loss(y_true, y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    huber = tf.keras.losses.Huber()(y_true, y_pred)
    rmse = tf.sqrt(tf.keras.losses.MeanSquaredError()(y_true, y_pred))
    #return 0.3 * mae + 0.3 * huber + 0.4 * rmse
    return 0.6 * mae + 0.4 * rmse
    #return rmse
    
## Load the model with custom objects
transunet_model = load_model(os.path.join(path_exp + 'transunet_model-regressor.keras'), custom_objects={'bounded_relu': bounded_relu, 'custom_loss': custom_loss})

# Predict LAI values
y_pred = transunet_model.predict(X_test_reshaped)
y_pred = np.round(y_pred, decimals=3)

print("Shape of X_test_combined_reshaped:", X_test_reshaped.shape)
print("Shape of y_pred:", y_pred.shape)
print("Shape of y_test:", y_test.shape)

y_pred = y_pred.squeeze()
print("Shape of y_pred after squeezing:", y_pred.shape)

# Ensure all have the same length
min_size = min(len(df_['x']), len(df_['y']), len(y_pred), len(y_test))

x_coords = df_['x'][:min_size]
y_coords = df_['y'][:min_size]
predicted_lai = y_pred[:min_size]
lai = y_test[:min_size]
plot = df_['Plot'][:min_size]

# Save the FID, x, y, predicted LAI, and S2 LAI to a CSV file
dfpredictions = pd.DataFrame({
#    'FID': dftest['FID'][:min_size],
    'Plot': plot,
    'LAI_pred': predicted_lai,
    'LAI_S2': lai,
    'x': x_coords,
    'y': y_coords,
    'date': df_['date'],
    'class': df_['class'][:min_size],
    'class_id': df_['class_id'][:min_size]
})

output_filename = os.path.join(path_val, 'satunet_predicted_2023.csv')
dfpredictions.to_csv(output_filename, index=False)
print(f"Predicted LAI vs S2 LAI saved to: {output_filename}")

# Save all columns along with predicted LAI values
df_['LAI_pred'] = y_pred[:len(df_)]  # Ensure alignment with dataframe length

# Include all the columns in the original dataframe along with the predicted LAI
output_filename = os.path.join(path_val, 'satunet_predicted_with_all_features_2023.csv')
df_.to_csv(output_filename, index=False)
print(f"Dataset with predicted LAI saved to: {output_filename}")
