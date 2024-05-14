import torch
import tifffile
import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
import numpy as np
from torchvision import transforms

image = tifffile.imread("testimage/HM25_HighRes_Aligned0050(1).tif")
image = (
    transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(image))
    .unsqueeze(0)
    .float()
    .numpy()
)

img_array = Array(
    image, roi=daisy.Roi((0, 0), (8000, 8000)), voxel_size=Coordinate(1, 1)
)

out_array = prepare_ds(
    "full_pred_example.zarr",
    "test1",
    total_roi=img_array.roi,
    voxel_size=(1, 1),
    dtype=np.float32,
    num_channels=2,
)


def process_block():
    client = daisy.Client()
    unet = torch.load("unet_mito.pt")

    while True:

        with client.acquire_block() as block:
            with torch.no_grad():
                image = torch.from_numpy(
                    img_array.to_ndarray(block.read_roi, fill_value=0)
                ).cuda()
                prediction = unet(image)
                prediction = Array(
                    prediction[0].cpu().numpy(), roi=block.read_roi, voxel_size=(1, 1)
                ).to_ndarray(block.write_roi)

                out_array[block.write_roi] = prediction


context = Coordinate(16, 16)
read_roi = Roi((0, 0), (800, 800)).grow(context, context)
write_roi = Roi((0, 0), (800, 800))

task = daisy.Task(
    "full_pred",
    total_roi=img_array.roi.grow(context, context),
    read_roi=read_roi,
    write_roi=write_roi,
    process_function=process_block,
    num_workers=1,
)

daisy.run_blockwise([task])
