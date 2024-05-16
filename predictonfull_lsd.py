import torch
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
import numpy as np
from torchvision import transforms
import zarr

import logging


raw_array = open_ds("test_data.zarr", "raw")


affs_array = prepare_ds(
    "full_pred_3d_ld.zarr",
    "affs",
    total_roi=raw_array.roi,
    voxel_size=raw_array.voxel_size,
    dtype=np.float32,
    num_channels=2,
)
lsd_array = prepare_ds(
    "full_pred_3d_ld.zarr",
    "lsds",
    total_roi=raw_array.roi,
    voxel_size=raw_array.voxel_size,
    dtype=np.float32,
    num_channels=6,
)


def process_block():
    client = daisy.Client()
    unet = torch.load("weights/pokemon-lsd-unet-ld.pt")

    while True:

        with client.acquire_block() as block:
            with torch.no_grad():

                # READ THE IMAGE DATA
                image = raw_array.to_ndarray(block.read_roi, fill_value=0)[
                    0
                ]  # (8000, 8000)
                image = transforms.Normalize([0.5], [0.5])(
                    transforms.ToTensor()(image)
                ).unsqueeze(0).cuda()

                logging.error(f"input range: ({image.min()}, {image.max()})")

                # PREDICT ON GPU
                prediction = unet(image)  # (1, 8, 8000, 8000)

                logging.error((prediction.min(), prediction.max()))

                # WRITE THE PREDICTION TO ZARR
                affs_pred, lsd_pred = prediction[:, :2, :, :], prediction[:, 2:, :, :]

                affs_pred = Array(
                    affs_pred.cpu().numpy().transpose([1, 0, 2, 3]),
                    roi=block.read_roi,
                    voxel_size=raw_array.voxel_size,
                ).to_ndarray(block.write_roi)

                affs = np.clip(affs_pred * 255, 0, 255).astype(np.uint8)
                logging.error(
                    (affs.min(), affs.max(), affs_pred.min(), affs_pred.max())
                )

                affs_array[block.write_roi] = affs

                lsd_pred = Array(
                    lsd_pred.cpu().numpy().transpose([1, 0, 2, 3]),
                    roi=block.read_roi,
                    voxel_size=raw_array.voxel_size,
                ).to_ndarray(block.write_roi)

                lsd_array[block.write_roi] = np.clip(lsd_pred * 255, 0, 255).astype(
                    np.uint8
                )


voxel_size = raw_array.voxel_size
context = Coordinate(0, 16, 16) * voxel_size
read_roi = Roi((0, 0, 0), (1, 800, 800)).grow(context, context) * voxel_size
write_roi = Roi((0, 0, 0), (1, 800, 800)) * voxel_size

task = daisy.Task(
    "full_pred_lsd_ld",
    total_roi=raw_array.roi.grow(context, context),
    read_roi=read_roi,
    write_roi=write_roi,
    process_function=process_block,
    num_workers=1,
)

daisy.run_blockwise([task])
