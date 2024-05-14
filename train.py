from matplotlib.colors import ListedColormap
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import distance_transform_edt
from local import train, plot_two, plot_three, plot_four
from unet import UNet
from tqdm import tqdm
import tifffile
from skimage.filters import threshold_otsu

#import dataset and model
#from dataset import ....
device = "cuda"  # 'cuda', 'cpu', 'mps'
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()




#get dataloader

learning_rate = 1e-4
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
epoch = 30

for epoch in range(epoch):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=10,
        device=device,
    )


val_data = #get validation dataset
unet.eval()
idx = np.random.randint(len(val_data))  # take a random sample. or 
image, sdt = val_data[idx]  # get the image and the nuclei masks.
image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))
image = np.squeeze(image.cpu())
sdt = np.squeeze(sdt.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())
plot_three(image, sdt, pred)


idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

# Hint: make sure set the model to evaluation
unet.eval()

image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))

image = np.squeeze(image.cpu())
mask = np.squeeze(mask.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())

# Choose a threshold value to use to get the boundary mask.
# Feel free to play around with the threshold.
threshold = threshold_otsu(pred)
print(f"Foreground threshold is {threshold:.3f}")

# Get inner mask
inner_mask = get_inner_mask(pred, threshold=threshold)

# Get the segmentation
seg = watershed_from_boundary_distance(pred, inner_mask, min_seed_distance=20)