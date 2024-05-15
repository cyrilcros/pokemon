import zarr
import tifffile
import matplotlib.pyplot as plt

predld = zarr.open('full_pred_example_ld.zarr')['lsds'][:]
# predmito = zarr.open('full_pred_example.zarr')['lsds'][:]
img = tifffile.imread("testimage/HM25_HighRes_Aligned0050(1).tif")

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img, cmap='gray')
# ax[1].imshow(predmito[0], cmap='summer', alpha=0.9)
# ax[1].imshow(predmito[1], cmap='gray', alpha=0.5)
ax[2].imshow(predld[0], cmap='summer', alpha=0.9)
plt.show()