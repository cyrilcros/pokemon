import daisy
from funlib.persistence import open_ds, prepare_ds, Array
from funlib.geometry import Coordinate, Roi
import numpy as np

from evaluate import threshold_otsu, distance_transform_edt, watershed_from_boundary_distance

import logging

affs_array = open_ds("full_pred_3d_ld.zarr", "affs")


frags_array = prepare_ds(
    "full_pred_3d_ld.zarr",
    "frags",
    total_roi=affs_array.roi,
    voxel_size=affs_array.voxel_size,
    dtype=np.uint64,
    num_channels=None,
)


def process_block():
    client = daisy.Client()

    while True:

        with client.acquire_block() as block:

            if block is None:
                break

            # READ THE IMAGE DATA
            affs = affs_array.to_ndarray(block.read_roi, fill_value=0)


            thresh = threshold_otsu(affs)
            # get boundary mask
            inner_mask = 0.5 * (affs[0] + affs[1]) > thresh
            boundary_distances = distance_transform_edt(inner_mask, sampling=affs_array.voxel_size)
            pred_labels = watershed_from_boundary_distance(
                boundary_distances, inner_mask, id_offset=0, min_seed_distance=50
            )

            offset = block.block_id[1] * np.prod(block.write_roi.shape / affs_array.voxel_size)
            pred_labels += (pred_labels > 0) * offset
            frags_pred = Array(
                pred_labels,
                roi=block.read_roi,
                voxel_size=affs_array.voxel_size,
            ).to_ndarray(block.write_roi)

            frags_array[block.write_roi] = frags_pred


voxel_size = affs_array.voxel_size
context = Coordinate(0, 16, 16) * voxel_size
read_roi = Roi((0, 0, 0), (10, 800, 800)).grow(context, context) * voxel_size
write_roi = Roi((0, 0, 0), (10, 800, 800)) * voxel_size

task = daisy.Task(
    "full_postprocess_lsd_ld",
    total_roi=affs_array.roi.grow(context, context),
    read_roi=read_roi,
    write_roi=write_roi,
    process_function=process_block,
    num_workers=8,
)

daisy.run_blockwise([task])
