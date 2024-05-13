from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class EMDataset(Dataset):
    """ Dataloader class for Chad's electron microscopy data
    Extends torch.utils.data.Dataset
    """
    def __init__(self,root_dir: str, category:str, transform=None, img_transform=None) -> None:
        """Initialization

        Args:
            root_dir (str): A train / validattion / test folder with images and masks subfolder.
            category (str): An organelle to segment.
            transform (_type_, optional): Transformation to apply to both images and masks. Defaults to None.
            img_transform (_type_, optional): Transformation that applies to masks only. Defaults to None.

        Raises:
            ValueError: If category is not one 'mito', 'ld', 'nucleus'.
        """
        super().__init__()
        # check the categories
        allowed_categories = ['mito', 'ld', 'nucleus']
        if category not in allowed_categories:
            raise ValueError(f"You asked for an item of category {category} which" +
                            f" is not one of {'/'.join(allowed_categories)}")
        self.category = category
        self.root_dir = root_dir
        # check the folder structure
        folders = os.listdir(root_dir)
        if 'images' not in folders or 'masks' not in folders:
            raise ValueError(f"I should have 'images' and 'masks' subfolders in {root_dir}")
        img_dir = os.path.join(self.root_dir, 'images')
        mask_dir = os.path.join(self.root_dir, 'masks')
        self.samples = os.listdir(img_dir)  # list the samples
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )
        #  transformations to apply just to inputs
        self.loaded_imgs = [None] * len(self.samples)
        self.loaded_masks = [None] * len(self.samples)
        for sample_ind, image_file in enumerate(self.samples):
            img_path = os.path.join(
               img_dir, image_file
            )
            image = Image.open(img_path)
            image.load()
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(
                mask_dir, image_file[:-4] + '_' + self.category + '.tif'
            )
            mask = Image.open(mask_path)
            mask.load()
            self.loaded_masks[sample_ind] = transforms.ToTensor()(mask)

   # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask
    
def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()
   
if __name__ == '__main__':
    train_dataset = EMDataset(root_dir='train', category='nucleus')
    print(f"Loaded a dataset {train_dataset}")
    show_random_dataset_image(train_dataset)