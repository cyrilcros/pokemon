import torch
from dataset import EMDataset
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed
from scipy.ndimage import label, maximum_filter
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
from matplotlib import ticker, gridspec
from model import UNet
from torchvision.transforms import v2
from collections import defaultdict

def find_local_maxima(distance_transform, min_dist_between_points):
    # Use `maximum_filter` to perform a maximum filter convolution on the distance_transform
    max_filtered = maximum_filter(distance_transform, min_dist_between_points)
    maxima = max_filtered == distance_transform
    # Uniquely label the local maxima
    seeds, n = label(maxima)
    return seeds, n

def plot_four(
    image: np.ndarray,
    intermediate: np.ndarray,
    pred: np.ndarray,
    seg: np.ndarray,
    label: str = "Target",
    cmap: str = "nipy_spectral",
):
    """
    Helper function to plot an image, the auxiliary (intermediate)
    representation of the target, the model prediction and the predicted segmentation mask.
    """
    fig = plt.figure(constrained_layout=False, figsize=(10, 3))
    spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.imshow(image)  # show the image
    ax1.set_xlabel("Image", fontsize=20)
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.imshow(intermediate)  # show the masks
    ax2.set_xlabel(label, fontsize=20)
    ax3 = fig.add_subplot(spec[0, 2])
    t = ax3.imshow(pred)
    ax3.set_xlabel("Pred.", fontsize=20)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar = fig.colorbar(t, fraction=0.046, pad=0.04)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax4 = fig.add_subplot(spec[0, 3])
    ax4.imshow(seg, cmap=cmap, interpolation="none")
    ax4.set_xlabel("Seg.", fontsize=20)
    _ = [ax.set_xticks([]) for ax in [ax1, ax2, ax3, ax4]]  # remove the xticks
    _ = [ax.set_yticks([]) for ax in [ax1, ax2, ax3, ax4]]  # remove the yticks
    plt.tight_layout()
    plt.show()

def watershed_from_boundary_distance(
    boundary_distances: np.ndarray,
    inner_mask: np.ndarray,
    id_offset: float = 0,
    min_seed_distance: int = 10,
):
    """Function to compute a watershed from boundary distances."""
    seeds, n = find_local_maxima(boundary_distances, min_seed_distance)
    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset
    seeds[seeds != 0] += id_offset
    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=inner_mask
    )
    return segmentation

def generate_labels(pred: np.ndarray) -> np.array: 
    """ Instance segmentation via watershed postprocessing

    Args:
        pred (np.array): Predictions with affinities / LSDs whatever

    Returns:
        np.array: Segmented instance
    """
    pred = np.squeeze(pred)
    # feel free to try different thresholds
    thresh = threshold_otsu(pred)
    # get boundary mask
    inner_mask = 0.5 * (pred[0] + pred[1]) > thresh
    boundary_distances = distance_transform_edt(inner_mask)
    pred_labels = watershed_from_boundary_distance(
        boundary_distances, inner_mask, id_offset=0, min_seed_distance=20
    )
    return pred_labels

def pixel_overlap_measure(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict[str: int] :
    """Look at overlap in non background pixel

    Returns:
        dict[str:int]: True/False Positive/Negative integer values
    """
    pred_pixel = np.sum(pred_labels > 0)
    gt_pixel = np.sum(gt_labels > 0)
    union_pixels = np.sum((pred_labels + gt_labels) > 0)
    common_pixels = gt_pixel + pred_pixel - union_pixels
    pixel_dict = {
        'tp': common_pixels,
        'fp': pred_pixel - common_pixels,
        'fn': gt_pixel - common_pixels,
    }
    return pixel_dict

def instance_overlap_measure(pred_labels: np.ndarray, gt_labels: np.ndarray, th: float = 0.5) -> dict[str: int] :
    """Look at overlap in distinct instances

    Returns:
        dict[str:int]: True/False Positive/Negative integer values
    """
    overlay = np.array([pred_labels.flatten(), gt_labels.flatten()])
    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1
    )
    overlay_labels = np.transpose(overlay_labels)
    gt_labels_list, gt_counts = np.unique(gt_labels, return_counts=True)
    gt_labels_count_dict = {}
    for l, c in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c
    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels, return_counts=True)
    pred_labels_count_dict = {}
    for l, c in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c
    num_pred_labels = int(np.max(pred_labels))
    num_gt_labels = int(np.max(gt_labels))
    num_matches = min(num_gt_labels, num_pred_labels)
    # create iou table
    iouMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32)
    for (u, v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)
        iouMat[int(v), int(u)] = iou
    # remove background
    iouMat = iouMat[1:, 1:]
    # use IoU threshold th
    if num_matches > 0 and np.max(iouMat) > th:
        costs = -(iouMat > th).astype(float) - iouMat / (2 * num_matches)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = iouMat[gt_ind, pred_ind] > th
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    fp = num_pred_labels - tp
    fn = num_gt_labels - tp
    return {'tp': tp, 'fp': fp, 'fn': fn}

def evaluate(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str: dict[str: int]]:
    """Will compute various measures comparing ground truth and predicted segmentations

    Returns:
        dict[str: dict[str: int]] Associates a measurement name with a dict of true/false
        positive/negative
    """
    # values to be returned
    measures = {}
    # computation
    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels.astype(np.uint8))
    measures['pixels_overlap'] = pixel_overlap_measure(pred_labels_rel, gt_labels_rel)
    measures['instances_overlap'] = instance_overlap_measure(pred_labels_rel, gt_labels_rel)
    return measures

def reduce_multiple_measures(measure_dict_list: list[dict]) -> dict:
    """_summary_

    Args:
        measure_dict_list (list[dict]): _description_

    Returns:
        dict: _description_
    """
    measures_merged = defaultdict(int)
    for measure_dict in measure_dict_list:
        for key, val in measure_dict.items():
            measures_merged[key] += val
    return measures_merged

def computed_metrics(measures: dict[str: dict[str: int]], epsilon = 1e-3) -> dict[str: dict[str: int]]:
    """_summary_

    Returns:
        _type_: _description_
    """
    metrics = {
        'precision': lambda vals : vals['tp'] / max(1, vals['tp'] + vals['fp']),
        'recall': lambda vals : vals['tp'] / max(1, vals['tp'] + vals['fn']),
        'accuracy': lambda vals : vals['tp'] / (vals['tp'] + vals['fp'] + vals['fn']),
        'F1-score': lambda vals : 2*vals['tp'] / (2*vals['tp'] + vals['fp'] + vals['fn']),
    }
    metric_results = {name_meas: {name_f: func(val_meas) for name_f, func in metrics.items()} for name_meas, val_meas in measures.items()}
    return metric_results

def run_eval(organelle: str, device: str, unet: UNet, stats_power = 100, patch_size=256) -> dict:
    # category is one of 'mito', 'ld', 'nucleus'
    # return mask must be true here!
    unet.eval()
    collected_metrics = []
    val_dataset = EMDataset(root_dir='validate/', category=organelle, 
                            return_mask=True, transform=v2.RandomCrop(patch_size))
    val_sampler = RandomSampler(data_source=val_dataset, replacement=True, num_samples=stats_power+1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, sampler=val_sampler)
    for loaded_vals in val_dataloader:
        if accepted_validation_images > stats_power:
            break
        mask = loaded_vals.pop()
        gt_labels = np.squeeze(mask.cpu().numpy())
        accepted_validation_images += 1
        image, _ = loaded_vals
        image = image.to(device).unsqueeze(0)
        with torch.no_grad():
            pred = unet(image)
        pred = pred.cpu().detach().numpy()
        pred_labels = generate_labels(pred)
        measures_instance = evaluate(gt_labels, pred_labels)
        collected_metrics.append(measures_instance)
    reduced_metrics = reduce_multiple_measures(collected_metrics)
    computed_metrics = pass

if __name__ == '__main__':
    organelle = 'ld'
    device = "cuda"  # 'cuda', 'cpu', 'mps'
    assert torch.cuda.is_available()
    model_name = f"pokemon-unet-{organelle}"
    unet = torch.load(f=f"weights/{model_name}.pt")
    metrics_avg = run_eval(organelle, device, unet)
    for measurement in metrics_avg:
        for val_name,val in metrics_avg[measurement].items():
            print(f"{measurement} {val_name} for {organelle} with model {model_name} is {val:.3f}")