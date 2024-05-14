import tifffile
import numpy as np  
from model import UNet
import torch
from matplotlib import pyplot as plt


unet = UNet(
    depth=4,
    in_channels=1,
    out_channels=1,
    final_activation="Tanh",
    num_fmaps=64,
    fmap_inc_factor=3,
    downsample_factor=2,
    padding="same",
    upsample_mode="nearest",
)

learning_rate = 1e-4
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

image_modeltest = tifffile.imread('/localscratch/pokemon/image_modeltest/HM25_HighRes_Aligned_0000_004-_cropped1k0040_0000.tif')
mask_modeltest = tifffile.imread('/localscratch/pokemon/mask_modeltest/HM25_HighRes_Aligned_0000_004-_cropped1k0040_0000_mito.tif')


tensor = torch.tensor(image_modeltest, dtype=torch.float32)
tensor = tensor.unsqueeze(0).unsqueeze(0)
# raise Exception(f"Tensor has shape: {tensor.shape}")

pred = unet(tensor)
tifffile.imwrite("tessssssssssssssssst.tiff", pred.detach().numpy().squeeze(0).squeeze(0))
plt.imshow(pred.detach().numpy().squeeze(0).squeeze(0))
plt.show()
print('prediction')