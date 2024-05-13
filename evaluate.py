from matplotlib.colors import ListedColormap
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import distance_transform_edt
from local import train, NucleiDataset, plot_two, plot_three, plot_four
from unet import UNet
from tqdm import tqdm
import tifffile
from skimage.filters import threshold_otsu

#import dataset and model
#from dataset import ....
device = "cuda"  # 'cuda', 'cpu', 'mps'
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()

