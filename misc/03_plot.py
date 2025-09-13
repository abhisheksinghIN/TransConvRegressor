import matplotlib.pyplot as plt
import rasterio
import numpy as np
import os
from datetime import datetime

# Define the folder containing the raster images
raster_folder = "/workspace/model-2023/satunet/proposed/LAI/Pred_LAI-FS/F1/"
raster_files = sorted([
    f for f in os.listdir(raster_folder) 
    if f.endswith(".tif") and f.split("_")[-1].split(".")[0].isdigit()
])

# Filter files for dates between May and September
filtered_files = []
for f in raster_files:
    date_str = f.split("_")[-1].split(".")[0]
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        if 3 <= date_obj.month <= 11:
            filtered_files.append(f)
    except ValueError:
        continue

# One row layout
num_images = len(filtered_files)
rows = 1
cols = num_images

# Adjust figure size
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, 3.5))  # Slightly larger for font

# Make axes iterable
if num_images == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# Global color scale
global_min = float("inf")
global_max = float("-inf")

for raster in filtered_files:
    with rasterio.open(os.path.join(raster_folder, raster)) as src:
        data = src.read(1)
        global_min = min(global_min, np.nanmin(data))
        global_max = max(global_max, np.nanmax(data))

# Plotting
for i, raster in enumerate(filtered_files):
    with rasterio.open(os.path.join(raster_folder, raster)) as src:
        data = src.read(1)
        im = axes[i].imshow(data, cmap="summer", vmin=0, vmax=8)
        date_str = raster.split("_")[-1].split(".")[0]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        formatted_date = date_obj.strftime("%d-%m-%Y")
        axes[i].text(0.5, -0.1, formatted_date, fontsize=18, ha="center", transform=axes[i].transAxes)
        axes[i].axis("off")


# Colorbar
cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.05, pad=0.02)
cbar.set_label("Range of LAI", fontsize=20)
cbar.set_ticks(np.linspace(0, 8, 9))

# Save
plot_path = os.path.join(raster_folder, "LAI_Combined_MaySep1.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"plot with larger font saved at: {plot_path}")
