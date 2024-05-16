import numpy as np
import napari

import zarr

container = zarr.open("test_data.zarr")
raw = container["raw"]

out_container = zarr.open("full_pred_3d_ld.zarr")
affs = out_container["affs"]
lsds = out_container["lsds"]
frags = out_container["frags"]


# Start a napari viewer
viewer = napari.Viewer()

raw = raw[:, 3000:5000, 3000:5000]
# affs = affs[:, :, 3000:5000, 3000:5000]
# lsds = lsds[:, :, 3000:5000, 3000:5000]
frags = frags[:, 3000:5000, 3000:5000]

# print(affs.min(), affs.max())
# print(lsds.min(), lsds.max())

# Add the 3D array to the viewer as an image layer
viewer.add_image(raw, name='Raw')
# viewer.add_image(affs, name='affs')
# viewer.add_image(lsds, name='lsds')
viewer.add_image(frags, name='frags')

# Start the napari event loop
napari.run()