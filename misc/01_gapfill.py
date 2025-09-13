"""
Description: 
    Converts tabular LAI predictions into GeoTIFF rasters for each plot and date.
    Saves outputs with consistent legend ranges and organized folders.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# -------------------------------
# CONFIGURATION
# -------------------------------
BASE_PATH = "./workspace/LAI/"
CSV_FILE = os.path.join(BASE_PATH, "LAI_S2_joined_nearest.csv")
OUTPUT_FOLDER = os.path.join(BASE_PATH, "S2_LAI-FS/F1N/")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PIXEL_SIZE = 10        # meters
CRS_EPSG = 32632       # UTM Zone 32N
PLOT_NAME = "F1"       # plot to process
MIN_ROWS_PER_DATE = 1  # minimum rows per date

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(CSV_FILE, sep=',')
required_cols = {'x', 'y', 'LAI_pred', 'S2_LAI', 'date', 'Plot_x'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Convert date column
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Filter for valid day numbers (optional)
df_filtered = df[df['date'].dt.day.isin(range(1, 32))]

# Filter for specific plot
df_plot = df_filtered[df_filtered['Plot_x'] == PLOT_NAME]

# Keep only dates with enough rows
date_counts = df_plot['date'].value_counts()
valid_dates = date_counts[date_counts > MIN_ROWS_PER_DATE].index
unique_dates = df_plot[df_plot['date'].isin(valid_dates)]['date'].unique()

# Compute global legend range
global_min = df_filtered['S2_LAI'].min()
global_max = df_filtered['S2_LAI'].max()
print(f"Global legend: LAI Min = {global_min}, Max = {global_max}")

# -------------------------------
# PROCESS EACH DATE
# -------------------------------
for date in unique_dates:
    df_date = df_plot[df_plot['date'] == date]

    # Keep numeric columns and drop NaNs
    df_numeric = df_date[['x', 'y', 'S2_LAI']].copy()
    df_numeric['S2_LAI'] = pd.to_numeric(df_numeric['S2_LAI'], errors='coerce')
    df_clean = df_numeric.dropna()

    # Aggregate duplicate coordinates
    df_grouped = df_clean.groupby(['x', 'y']).mean().reset_index()

    # Define raster extent
    x_min, x_max = df_grouped['x'].min(), df_grouped['x'].max()
    y_min, y_max = df_grouped['y'].min(), df_grouped['y'].max()
    cols = int((x_max - x_min) / PIXEL_SIZE) + 1
    rows = int((y_max - y_min) / PIXEL_SIZE) + 1

    # Initialize raster
    raster = np.full((rows, cols), np.nan)

    # Map coordinates to raster indices
    df_grouped['row'] = np.clip(((y_max - df_grouped['y']) / PIXEL_SIZE).astype(int), 0, rows - 1)
    df_grouped['col'] = np.clip(((df_grouped['x'] - x_min) / PIXEL_SIZE).astype(int), 0, cols - 1)
    df_grouped = df_grouped[(df_grouped['row'] >= 0) & (df_grouped['row'] < rows)]
    df_grouped = df_grouped[(df_grouped['col'] >= 0) & (df_grouped['col'] < cols)]

    # Fill raster with values
    for _, r in df_grouped.iterrows():
        raster[int(r['row']), int(r['col'])] = r['S2_LAI']

    # Raster metadata
    transform = from_origin(x_min, y_max, PIXEL_SIZE, PIXEL_SIZE)
    crs = CRS.from_epsg(CRS_EPSG)

    # Output folder per plot
    plot_folder = os.path.join(OUTPUT_FOLDER, PLOT_NAME)
    os.makedirs(plot_folder, exist_ok=True)
    output_path = os.path.join(plot_folder, f"LAI_pred_{date.strftime('%Y%m%d')}.tif")

    # Save GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=raster.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(raster, 1)
        dst.update_tags(legend_min=float(global_min), legend_max=float(global_max),
                        description="LAI Prediction Range")

    print(f"Saved GeoTIFF for Plot '{PLOT_NAME}' on {date.strftime('%Y-%m-%d')}: {output_path}")
