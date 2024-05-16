import os
import torch
from torch.utils.data import DataLoader
from model import unet
from dataset import EMDataset, transform_to_do
from torch.utils.tensorboard import SummaryWriter
from evaluate import run_eval
import tqdm

# category is one of 'mito', 'ld', 'nucleus'
organelle = 'nucleus'

def train(model, organelle, loader, optimizer, loss_function, epoch, log_image_interval = 20,
          log_validate_interval = 40, tb_logger=None, device=None, early_stop=False):
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
    for batch_id, (x, y) in enumerate(loader):
        # zero the gradients for this iteration
        optimizer.zero_grad()
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        # apply model and calculate loss
        prediction = model(x)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)
        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()
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
                    img_tensor = (0.5 * img_actual[:,0] + 0.5*img_actual[:,1]).unsqueeze(dim=1),
                    global_step=step
                )
                img_pred = prediction.to("cpu").detach()
                tb_logger.add_images(
                    tag="pred",  
                    img_tensor = (0.5 * img_pred[:,0] + 0.5*img_pred[:,1]).unsqueeze(dim=1),
                    global_step=step
                )
            if step % log_validate_interval == 0:
                _, avg_metrics = run_eval(organelle=organelle, device=device, unet=model)
                for measure_name, metric_dict in avg_metrics.items():
                    for metric_name in metric_dict.keys():
                        tb_logger.add_scalar(
                            tag=f"{measure_name}_{metric_name}", scalar_value=avg_metrics[measure_name][metric_name], global_step=step
                        )
                model.train()

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break

#import dataset and model
#from dataset import ....
device = "cuda"  # 'cuda', 'cpu', 'mps'
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()

model_name = f"pokemon-unet-{organelle}"

# returns image 1000x1000, affinity 2x1000x1000, and if return_mask a mask 1000x1000
train_dataset = EMDataset(root_dir='train', category=organelle, return_mask=False, transform=transform_to_do)

# dataloader from train dataset
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=8)

# Logger
logger = SummaryWriter(f"runs/{model_name}")

learning_rate = 1e-4
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1e-4, total_iters=30)
epoch = 100

for epoch in tqdm.tqdm(range(epoch)):
    train(
        model = unet,
        organelle=organelle,
        loader=train_loader,
        optimizer=optimizer,
        loss_function=loss,
        epoch=epoch,
        tb_logger=logger,
        log_image_interval=20,
        log_validate_interval=40,
        device=device
    )
    scheduler.step()

weight_folder = 'weights'
if not os.path.exists(weight_folder):
    os.mkdir(weight_folder)
torch.save(unet, f=f"{weight_folder}/{model_name}.pt")