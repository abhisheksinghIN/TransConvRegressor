import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from time import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, PredictionErrorDisplay
from matplotlib.ticker import MultipleLocator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from network import TransConvRegressor, custom_loss, bounded_relu
# --------------------------- Paths ---------------------------
path = "./workspace/"
filename_database = path + "database_2023-SM.csv"
path_exp = "./workspace/"
os.makedirs(path_exp, exist_ok=True)

# --------------------------- Load Data ---------------------------
# Note: this follows your original script closely and preserves the dropped columns
df_ = pd.read_csv(filename_database, sep='\t').drop([
    "Unnamed: 0", 'RTC_dsc_RVI', 'RTC_dsc_ratio', 'RTC_dsc_sum', 'RTC_dsc_diff', 'RTC_dsc_prod',
    'RTC_dsc_VH_prod', 'RTC_dsc_sum_prod', 'RTC_dsc_square_diff', 'RTC_asc_RVI',
    'RTC_asc_ratio', 'RTC_asc_sum', 'RTC_asc_diff', 'RTC_asc_prod', 'RTC_asc_VH_prod',
    'RTC_asc_sum_prod', 'RTC_asc_square_diff'
], axis=1)

# ************************************************** Preprocessing **************************************************
df_['doy_sin'] = np.sin(2 * np.pi * df_['doy'] / 365)
df_['doy_cos'] = np.cos(2 * np.pi * df_['doy'] / 365)
df_['classid_sin'] = np.sin(2 * np.pi * df_['class_id'] / 8)
df_['classid_cos'] = np.cos(2 * np.pi * df_['class_id'] / 8)

# Convert dB to linear for the radar bands
df_["RTC_dsc_VH"] = 10 ** (df_["RTC_dsc_VH"] * 0.1)
df_["RTC_dsc_VV"] = 10 ** (df_["RTC_dsc_VV"] * 0.1)
df_["RTC_asc_VH"] = 10 ** (df_["RTC_asc_VH"] * 0.1)
df_["RTC_asc_VV"] = 10 ** (df_["RTC_asc_VV"] * 0.1)

# Derived Features
df_["RTC_dsc_RVI"] = 4 * df_["RTC_dsc_VH"] / (df_["RTC_dsc_VV"] + df_["RTC_dsc_VH"])
df_["RTC_dsc_ratio"] = df_["RTC_dsc_VH"] / df_["RTC_dsc_VV"]

df_["RTC_asc_RVI"] = 4 * df_["RTC_asc_VH"] / (df_["RTC_asc_VV"] + df_["RTC_asc_VH"])
df_["RTC_asc_ratio"] = df_["RTC_asc_VH"] / df_["RTC_asc_VV"]

df_["RTC_dsc_prod"] = df_["RTC_dsc_VV"] * df_["RTC_dsc_VH"]
df_["RTC_asc_prod"] = df_["RTC_asc_VV"] * df_["RTC_asc_VH"]

df_["RTC_dsc_sum"] = df_["RTC_dsc_VV"] + df_["RTC_dsc_VH"]
df_["RTC_asc_sum"] = df_["RTC_asc_VV"] + df_["RTC_asc_VH"]

# --------------------------- Scaling ---------------------------
columns_to_scale = [
    "RTC_dsc_VV", "RTC_asc_VV", "RTC_dsc_VH", "RTC_asc_VH",
    'RTC_dsc_RVI','RTC_asc_RVI', 'RTC_dsc_ratio', 'RTC_asc_ratio',
    'RTC_dsc_prod', 'RTC_asc_prod', 'RTC_dsc_sum', 'RTC_asc_sum',
    'SM', 'doy_sin', 'doy_cos', 'classid_sin', 'classid_cos'
]
scaler = StandardScaler()
df_[columns_to_scale] = scaler.fit_transform(df_[columns_to_scale])
dump(scaler, os.path.join(path_exp, 'scaler.pkl'))
print("Scaler saved as 'scaler.pkl'")

# Feature and target selection
X_sequence = df_[[
    "RTC_dsc_VV", "RTC_asc_VV", "RTC_dsc_VH", "RTC_asc_VH",
    'RTC_dsc_RVI','RTC_asc_RVI', 'RTC_dsc_ratio', 'RTC_asc_ratio',
    'RTC_dsc_prod', 'RTC_asc_prod', 'RTC_dsc_sum', 'RTC_asc_sum','SM',
    'classid_sin', 'classid_cos', 'doy_sin', 'doy_cos'
]].values
Y = df_['LAI'].values  # Target variable

# ****************************** Stratification ******************************
bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Define bin edges
bin_labels = [1, 2, 3, 4, 5, 6, 7, 8]  # Label bins for stratification

# Create LAI bins
df_['LAI_bin'] = pd.cut(df_['LAI'], bins=bin_edges, labels=bin_labels, include_lowest=True)

# Check bin distribution
bin_counts = df_['LAI_bin'].value_counts()
print(f"Samples per bin:\n{bin_counts}")

#*************************************** Defining separate Test sets for Validation ****************************
# Preserve original behavior: add row_index and month grouping
df_['row_index'] = df_.index
df_['date'] = pd.to_datetime(df_['date'])
# Group by month or date
df_['month'] = df_['date'].dt.to_period('M')

split_indices = {'train': [], 'val': [], 'test': []}

for group_value, group_df in df_.groupby('month'):
    if len(group_df) < 2:
        continue
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.45, random_state=42)
    try:
        train_idx, temp_idx = next(sss1.split(group_df, group_df['LAI_bin']))
    except ValueError:
        continue

    df_temp = group_df.iloc[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=8/9, random_state=42)
    try:
        val_idx, test_idx = next(sss2.split(df_temp, df_temp['LAI_bin']))
    except ValueError:
        continue

    # Map back to original indices
    split_indices['train'].extend(group_df.iloc[train_idx].index)
    split_indices['val'].extend(group_df.iloc[temp_idx[val_idx]].index)
    split_indices['test'].extend(group_df.iloc[temp_idx[test_idx]].index)

# save split indices
 with open('./workspace/fixed_monthly_split_indices.pkl', 'wb') as f:
     pickle.dump(split_indices, f)

# Load indices if needed
fixed_split_file = './workspace/git/fixed_monthly_split_indices.pkl'
try:
    with open(fixed_split_file, 'rb') as f:
        split_indices = pickle.load(f)
    print(f"Loaded fixed split indices from {fixed_split_file}")
except FileNotFoundError:
    print(f"Fixed split file not found at {fixed_split_file}. Using computed split_indices.")

train_idx = split_indices['train']
val_idx = split_indices['val']
test_idx = split_indices['test']

# Create train/val/test DataFrames
df_train = df_.iloc[train_idx].reset_index(drop=True)
df_val   = df_.iloc[val_idx].reset_index(drop=True)
df_test  = df_.iloc[test_idx].reset_index(drop=True)

# *************************************** Prepare Input Features ***************************************
features = [
    "RTC_dsc_VV", "RTC_asc_VV", "RTC_dsc_VH", "RTC_asc_VH",
    'RTC_dsc_RVI','RTC_asc_RVI', 'RTC_dsc_ratio', 'RTC_asc_ratio',
    'RTC_dsc_prod', 'RTC_asc_prod', 'RTC_dsc_sum', 'RTC_asc_sum','SM',
    'classid_sin', 'classid_cos', 'doy_sin', 'doy_cos'
]

X_train = df_train[features].values
y_train = df_train['LAI'].values

X_val = df_val[features].values
y_val = df_val['LAI'].values

X_test = df_test[features].values
y_test = df_test['LAI'].values

# Reshape for CNN
def reshape_for_cnn(X):
    return X.reshape((X.shape[0], X.shape[1], 1))

X_train_reshaped = reshape_for_cnn(X_train)
X_val_reshaped = reshape_for_cnn(X_val)
X_test_reshaped = reshape_for_cnn(X_test)

#************************************************ Model ********************************************************
input_shape_combined = (X_train_reshaped.shape[1], 1)
# Create the TransConvRegressor model
TransConvRegressor_model = TransConvRegressor(input_shape_combined)
TransConvRegressor_model.compile(optimizer=Adam(learning_rate=0.0001), loss=custom_loss)

# Save model summary
summary_file = os.path.join(path_exp, "TransConvRegressor_model_summary.txt")
with open(summary_file, "w") as file:
    TransConvRegressor_model.summary(print_fn=lambda x: file.write(x + "\n"))
print(f"Model summary saved to {summary_file}")

# ---------------- Training ----------------
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

start_time = time()
history1 = TransConvRegressor_model.fit(
    X_train_reshaped, y_train,
    validation_data=(X_val_reshaped, y_val),
    epochs=1, batch_size=128, callbacks=[early_stopping]
)
elapsed_time = time() - start_time
elapsed_time = f"{elapsed_time:.3f}"

# Save trained model
TransConvRegressor_model.save(os.path.join(path_exp, 'TransConvRegressor_model-regressor.keras'))
print("TransConvRegressor_model saved as TransConvRegressor_model-regressor.keras")

# save training history to text file
history_file = os.path.join(path_exp, "TransConvRegressor-model_training_history.txt")
with open(history_file, "w") as file:
    file.write("Training History\n")
    file.write("=================\n")
    for epoch, loss in enumerate(history1.history['loss']):
        val_loss = history1.history['val_loss'][epoch] if 'val_loss' in history1.history else None
        file.write(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f if val_loss is not None else 'N/A'}\n")
print(f"Training history saved to {history_file}")

# ---------------- Evaluation ----------------
## Load the model with custom objects (demonstrates recovering later)
#TransConvRegressor_model = load_model(os.path.join(path_exp, 'TransConvRegressor_model-regressor.keras'), custom_objects={'bounded_relu': bounded_relu, 'custom_loss': custom_loss})
# Predict LAI values
y_pred = TransConvRegressor_model.predict(X_test_reshaped)
y_pred = y_pred.squeeze()

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
bias = np.mean(y_pred.flatten() - y_test.flatten())
mae = mean_absolute_error(y_test, y_pred)

# Print metrics to console
print(f"Elapsed Time: {elapsed_time}")
print(f"R² score: {r2}")
print(f"RMSE: {rmse}")
print(f"BIAS: {bias}")
print(f"MAE: {mae}")

# Save outcomes to a text file
output_file = os.path.join(path_exp, "model_evaluation_metrics.txt")
with open(output_file, "w") as file:
    file.write("Model Evaluation Metrics\n")
    file.write("=========================\n")
    file.write(f"Elapsed Time: {elapsed_time}\n")
    file.write(f"R² score: {r2}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"BIAS: {bias}\n")
    file.write(f"MAE: {mae}\n")

print(f"Metrics saved to {output_file}")

#*************************************************** Plots and Predicted Results **********************************************************
# Ensure all have the same length
min_size = min(len(df_test['x']), len(df_test['y']), len(y_pred), len(y_test))

x_coords = df_test['x'][:min_size]
y_coords = df_test['y'][:min_size]
predicted_lai = y_pred[:min_size]
lai = y_test[:min_size]

# Save the FID, x, y, predicted LAI, and S2 LAI to a CSV file
df_predictions = pd.DataFrame({
    'FID': df_test['FID'][:min_size],
    'LAI_pred': predicted_lai,
    'LAI_S2': lai,
    'x': x_coords,
    'y': y_coords,
    'date': df_test['date'][:min_size],
    'class': df_test['class'][:min_size],
    'row_index': df_test['row_index'][:min_size]
})

output_filename = os.path.join(path_exp, 'predicted_vs_s2_lai-date_CNN.csv')
df_predictions.to_csv(output_filename, index=False)
print(f"Predicted LAI vs S2 LAI saved to: {output_filename}")

#*************************************************** Scatter Plot **********************************************************
from matplotlib.ticker import MultipleLocator

FONT_SIZE = 20
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), dpi=200)

# ---- Plot 1: Observed vs Predicted ----
PredictionErrorDisplay.from_predictions(
    y_test, y_pred=y_pred,
    kind="actual_vs_predicted", subsample=1000, ax=axs[0], random_state=0
)
axs[0].plot([0, 8], [0, 8], 'k--', linewidth=1.5, label='1:1 line')
axs[0].set_title("Observed vs. Predicted", fontsize=FONT_SIZE)
axs[0].set_ylabel("Observed LAI", fontsize=FONT_SIZE)
axs[0].set_xlabel("Predicted LAI", fontsize=FONT_SIZE)
axs[0].set_xlim(0, 8)
axs[0].set_ylim(0, 8)
axs[0].set_xticks(np.arange(0, 8, 2))
axs[0].set_yticks(np.arange(0, 8, 2))
axs[0].tick_params(axis='both', labelsize=FONT_SIZE)
axs[0].xaxis.set_major_locator(MultipleLocator(1))
axs[0].yaxis.set_major_locator(MultipleLocator(1))
axs[0].grid(True, linestyle='--', alpha=0.4)

# ---- Plot 2: Residuals vs Predicted ----
residuals = y_test - y_pred
PredictionErrorDisplay.from_predictions(
    y_test, y_pred=y_pred,
    kind="residual_vs_predicted", subsample=1000, ax=axs[1], random_state=0
)
axs[1].axhline(0, color='k', linestyle='--', linewidth=1.5)
axs[1].set_title("Residuals vs. Predicted", fontsize=FONT_SIZE)
axs[1].set_xlabel("Predicted LAI", fontsize=FONT_SIZE)
axs[1].set_ylabel("Residual (Observed - Predicted)", fontsize=FONT_SIZE)
axs[1].set_xlim(0, 8)
axs[1].set_ylim(-3, 5)
axs[1].set_xticks(np.arange(0, 8, 2))
axs[1].set_yticks(np.arange(-3, 5, 2))
axs[1].tick_params(axis='both', labelsize=FONT_SIZE)
axs[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
fig.savefig(path_exp + 'LAI_Model_Performance_FS_validation_v2.png', dpi=1000)
plt.close()

#************************************************ Box Plot ***************************************************
FONT_SIZE = 24
boxplot_fig, box_ax = plt.subplots(figsize=(14, 6))

df_predictions['date'] = pd.to_datetime(df_predictions['date'])
df_predictions['month_str'] = df_predictions['date'].dt.strftime('%Y-%m')

df_melt = df_predictions[['month_str', 'LAI_S2', 'LAI_pred']].copy()
df_melt = df_melt.melt(id_vars='month_str', value_vars=['LAI_S2', 'LAI_pred'], var_name='Source', value_name='LAI')

sns.boxplot(data=df_melt, x='month_str', y='LAI', hue='Source', ax=box_ax, width=0.5, palette={'LAI_S2': 'green', 'LAI_pred': 'lightgreen'})

box_ax.set_ylabel('LAI', fontsize=FONT_SIZE)
box_ax.set_xlabel('Month', fontsize=FONT_SIZE)
box_ax.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE - 2)
box_ax.tick_params(axis='y', labelsize=FONT_SIZE - 2)

handles, labels = box_ax.get_legend_handles_labels()
box_ax.legend_.remove()

boxplot_file = os.path.join(path_exp, "satunet_boxplot_LAI_by_date_v1.png")
boxplot_fig.tight_layout()
boxplot_fig.savefig(boxplot_file, dpi=1000)
plt.close(boxplot_fig)
print(f"Date-wise boxplot saved to: {boxplot_file}")

# Save legend separately
legend_fig, legend_ax = plt.subplots(figsize=(4, 2))
legend = legend_ax.legend(handles=handles, labels=labels, title='Source', title_fontsize=FONT_SIZE, fontsize=FONT_SIZE)
legend_ax.axis('off')
legend_file = os.path.join(path_exp, "satunet_legend_only.png")
legend_fig.tight_layout()
legend_fig.savefig(legend_file, dpi=300, bbox_inches='tight', transparent=True)
plt.close(legend_fig)
print(f"Legend saved separately to: {legend_file}")

# Final additional boxplot (kept as in original)
boxplot_fig, box_ax = plt.subplots(figsize=(14, 6))
df_melt = df_predictions[['month_str', 'LAI_S2', 'LAI_pred']].copy()
df_melt = df_melt.melt(id_vars='month_str', value_vars=['LAI_S2', 'LAI_pred'], var_name='Source', value_name='LAI')

sns.boxplot(data=df_melt, x='month_str', y='LAI', hue='Source', ax=box_ax, width=0.5, palette={'LAI_S2': 'green', 'LAI_pred': 'lightgreen'})
box_ax.set_title(f'SATUNET - LAI Distribution by Date')
box_ax.set_ylabel('LAI')
box_ax.set_xlabel('Month')
box_ax.tick_params(axis='x', rotation=45)
box_ax.legend(title='Source')
boxplot_file = os.path.join(path_exp, f"satunet_boxplot_LAI_by_date_v1.png")
boxplot_fig.tight_layout()
boxplot_fig.savefig(boxplot_file, dpi=300)
print(f"Date-wise boxplot saved to {boxplot_file}")
plt.close(boxplot_fig)

