import zarr
import tifffile
from pathlib import Path
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
import numpy as np

tiff_path = "stack/stack/Tiffs0041-0050/HM25_HighRes_Aligned00{z}.tif"


stack_array = prepare_ds(
    "test_data.zarr",
    "raw",
    total_roi=Roi((410, 0, 0), (100, 8000, 8000)), # this uses the voxel size
    voxel_size=(10, 1, 1), # this voxel size is somewhat arbitrary, make real
    dtype=np.uint8,
    num_channels=None,
) # shape of this array is (10, 8000, 8000)

for z in range(41, 51):
    img = tifffile.imread(tiff_path.format(z=z))
    stack_array.data[z-41] = img