import zarr
import tifffile
import matplotlib.pyplot as plt

pred = zarr.open('full_pred_example.zarr')['nucleus'][:]
img = tifffile.imread("testimage/HM25_HighRes_Aligned0050(1).tif")

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(pred[0], cmap='Reds', alpha=0.5)
ax[1].imshow(pred[1], cmap='Greens', alpha=0.5)
plt.show()