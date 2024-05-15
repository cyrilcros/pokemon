from matplotlib.colors import ListedColormap
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from scipy.ndimage import distance_transform_edt
from model import lsd_unet
from tqdm import tqdm
from skimage.filters import threshold_otsu
from dataset_lsds import EMDataset
from torch.utils.tensorboard import SummaryWriter


def train(
    model,
    loader,
    optimizer,
    loss_function,
    lsd_loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y, lsd) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y, lsd = x.to(device), y.to(device), lsd.to(device)
        x = x.unsqueeze(1)
        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        # if prediction.shape != y.shape: #remove TODO
        #     y = crop(y, prediction)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)

        affs_pred, lsd_pred = prediction[:, :2, :, :], prediction[:, 2:, :, :]
        loss_affs = loss_function(affs_pred, y)
        loss_lsd = lsd_loss_function(lsd_pred, lsd)
        loss = loss_affs + loss_lsd

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\taffs Loss: {:.6f}, lsd Loss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss_affs.item(),
                    loss_lsd.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                img_actual = y.to("cpu")
                tb_logger.add_images(
                    tag="target",
                    img_tensor=(
                        0.5 * img_actual[:, 0] + 0.5 * img_actual[:, 1]
                    ).unsqueeze(dim=1),
                    global_step=step,
                )
                img_pred = prediction.to("cpu").detach()
                tb_logger.add_images(
                    tag="pred",
                    img_tensor=(0.5 * img_pred[:, 0] + 0.5 * img_pred[:, 1]).unsqueeze(
                        dim=1
                    ),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


# category is one of 'mito', 'ld', 'nucleus'
organelle = "mito"

# import dataset and model
# from dataset import ....
device = "cuda"  # 'cuda', 'cpu', 'mps'
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()

model_name = f"pokemon-lsd-unet-{organelle}"

# returns image 1000x1000, affinity 2x1000x1000, and if return_mask a mask 1000x1000
train_dataset = EMDataset(
    root_dir="train",
    category=organelle,
    return_mask=False,
    transform=transforms.Compose(
        [
            # transforms.RandomRotation(360),
            transforms.RandomCrop(256)]
    ),
)

# dataloader from train dataset
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=8)

# Logger
logger = SummaryWriter(f"runs/{model_name}")

learning_rate = 1e-4
loss = torch.nn.BCELoss()
lsd_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lsd_unet.parameters(), lr=learning_rate)
epoch = 300

for epoch in range(epoch):
    train(
        lsd_unet,
        train_loader,
        optimizer,
        loss,
        lsd_loss,
        epoch,
        tb_logger=logger,
        log_interval=10,
        device=device,
    )

weight_folder = "weights"
if not os.path.exists(weight_folder):
    os.mkdir(weight_folder)
torch.save(lsd_unet, f=f"{weight_folder}/{model_name}.pt")
