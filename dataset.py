from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
from PIL import Image

class EMDataset(Dataset):
   """A PyTorch dataset to load EM and masks"""
   def __init__(self,root_dir, seed: int, categories=["mito", "ld", "nucleus"], transform=None, img_transform=None) -> None:
       super().__init__()
       self.seed = seed
       self.categories = categories
       self.root_dir = root_dir
       self.samples = os.listdir(self.root_dir)  # list the samples
       self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
       self.img_transform = img_transform  # transformations to apply to raw image only
       #  transformations to apply just to inputs
       inp_transforms = transforms.Compose(
           [
               #transforms.Grayscale(),
               #transforms.ToTensor(),
               #transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
           ]
       )
       self.loaded_imgs = [None] * len(self.samples)
       self.loaded_masks = [None] * len(self.samples)
       for sample_ind in range(len(self.samples)):
           img_path = os.path.join(
               self.root_dir, self.samples[sample_ind], "image.tif"
           )
           image = Image(img_path)
           image.load()
           self.loaded_imgs[sample_ind] = inp_transforms(image)
        
           mask_path = os.path.join(
               self.root_dir, self.samples[sample_ind], "mask.tif"
           )
           mask = Image(mask_path)
           mask.load()
           self.loaded_masks[sample_ind] = transforms.ToTensor()(mask)

   # get the total number of samples
   def __len__(self):
       return len(self.samples)

   # fetch the training sample given its index
   def __getitem__(self, idx, category: str):
       # check the categories
       if category not in self.categories:
           raise R
       # we'll be using Pillow library for reading files
       # since many torchvision transforms operate on PIL images
       image = self.loaded_imgs[idx]
       mask = self.loaded_masks[idx]
       if self.transform is not None:
           # Note: using seeds to ensure the same random transform is applied to
           # the image and mask
           torch.manual_seed(self.seed)
           image = self.transform(image)
           torch.manual_seed(self.seed)
           mask = self.transform(mask)
       if self.img_transform is not None:
           image = self.img_transform(image)
       return image, mask
   
if __name__ == '__main__':
    train_dataset = EMDataset(root_dir='train')
    print(f"Loaded a dataset {train_dataset}")