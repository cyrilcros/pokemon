import torch
import tifffile
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
import numpy as np
from torchvision import transforms
import zarr

# image = tifffile.imread("testimage/HM25_HighRes_Aligned0050(1).tif")
image = zarr.open("test_data.zarr")["raw"][0, :, :]
image = (
    transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(image))
    .unsqueeze(0)
    .float()
    .numpy()
)

img_array = Array(
    image, roi=daisy.Roi((0, 0), (8000, 8000)), voxel_size=Coordinate(1, 1)
)

affs_array = prepare_ds(
    "full_pred_example_ld.zarr",
    "affs",
    total_roi=img_array.roi,
    voxel_size=(1, 1),
    dtype=np.float32,
    num_channels=2,
)
lsd_array = prepare_ds(
    "full_pred_example_ld.zarr",
    "lsds",
    total_roi=img_array.roi,
    voxel_size=(1, 1),
    dtype=np.float32,
    num_channels=6,
)


def process_block():
    client = daisy.Client()
    unet = torch.load("weights/pokemon-lsd-unet-ld.pt")

    while True:

        with client.acquire_block() as block:
            with torch.no_grad():
                image = torch.from_numpy(
                    img_array.to_ndarray(block.read_roi, fill_value=0)
                ).cuda()
                prediction = unet(image)
                affs_pred, lsd_pred = prediction[:, :2, :, :], prediction[:, 2:, :, :]

                affs_pred = Array(
                    affs_pred[0].cpu().numpy(), roi=block.read_roi, voxel_size=(1, 1)
                ).to_ndarray(block.write_roi)

                affs_array[block.write_roi] = affs_pred

                lsd_pred = Array(
                    lsd_pred[0].cpu().numpy(), roi=block.read_roi, voxel_size=(1, 1)
                ).to_ndarray(block.write_roi)

                lsd_array[block.write_roi] = lsd_pred


context = Coordinate(16, 16)
read_roi = Roi((0, 0), (800, 800)).grow(context, context)
write_roi = Roi((0, 0), (800, 800))

task = daisy.Task(
    "full_pred_lsd",
    total_roi=img_array.roi.grow(context, context),
    read_roi=read_roi,
    write_roi=write_roi,
    process_function=process_block,
    num_workers=1,
)

daisy.run_blockwise([task])
