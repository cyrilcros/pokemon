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

def evaluate(gt_labels: np.ndarray, pred_labels: np.ndarray, 
             th: float = 0.5, epsilon=1e-3) -> tuple[bool, dict]:
    """Function to evaluate a segmentation."""
    # values to be returned
    metrics = {}
    # computation
    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels.astype(np.uint8))
    overlay = np.array([pred_labels_rel.flatten(), gt_labels_rel.flatten()])
    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1
    )
    overlay_labels = np.transpose(overlay_labels)
    # look at total pixel overlap
    pred_pixel = np.sum(pred_labels_rel > 0)
    gt_pixel = np.sum(gt_labels_rel > 0)
    union_pixels = np.sum((pred_labels_rel + gt_labels_rel) > 0)
    common_pixels = gt_pixel + pred_pixel - union_pixels
    tp = common_pixels
    fp = pred_pixel - common_pixels
    fn = gt_pixel - common_pixels
    metrics['precision-pixel-overlap'] = tp / max(1, tp + fp)
    metrics['recall-pixel-overlap'] = tp / max(1, tp + fn)
    metrics['accuracy-pixel-overlap'] = tp / (tp + fp + fn)
    metrics['F1-pixel-overlap'] = 2*metrics['precision-pixel-overlap']*metrics['recall-pixel-overlap'] / \
        (metrics['recall-pixel-overlap']+metrics['precision-pixel-overlap']+epsilon)
    # TODO remove #print(f'common {common_pixels} pred {only_pred_pixel} gt {only_gt_pixel}')
    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    gt_labels_count_dict = {}
    for l, c in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c
    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels_rel, return_counts=True)
    pred_labels_count_dict = {}
    for l, c in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c
    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
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
    metrics['precision-detections'] = tp / max(1, tp + fp)
    metrics['recall-detections'] = tp / max(1, tp + fn)
    metrics['accuracy-detections'] = tp / (tp + fp + fn)
    metrics['F1-detections'] = 2*metrics['precision-detections']*metrics['recall-detections'] / \
        (metrics['recall-detections']+metrics['precision-detections']+epsilon)
    return num_gt_labels+num_pred_labels>0, metrics

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

def run_eval(organelle: str, device: str, unet: UNet, stats_power = 100, quit_thres = 300) -> dict:
    # category is one of 'mito', 'ld', 'nucleus'
    # return mask must be true here!
    val_dataset = EMDataset(root_dir='validate/', category=organelle, 
                            return_mask=True, transform=v2.RandomCrop(256))
    val_sampler = RandomSampler(data_source=val_dataset, replacement=True, num_samples=quit_thres+10)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, sampler=val_sampler)
    unet.eval()
    metrics_dict = defaultdict(list)
    accepted_validation_images, seen_images = 0, 0
    for loaded_vals in val_dataloader:
        if accepted_validation_images > stats_power:
            break
        if seen_images > quit_thres:
            print(f'Failed to get enough examples of {organelle} from random patches')
            break
        seen_images += 1
        mask = loaded_vals.pop()
        gt_labels = np.squeeze(mask.cpu().numpy())
        accepted_validation_images += 1
        image, _ = loaded_vals
        image = image.to(device).unsqueeze(0)
        with torch.no_grad():
            pred = unet(image)
        image = np.squeeze(image.cpu())
        pred = np.squeeze(pred.cpu().detach().numpy())
        # feel free to try different thresholds
        thresh = threshold_otsu(pred)
        # get boundary mask
        inner_mask = 0.5 * (pred[0] + pred[1]) > thresh
        boundary_distances = distance_transform_edt(inner_mask)
        pred_labels = watershed_from_boundary_distance(
            boundary_distances, inner_mask, id_offset=0, min_seed_distance=20
        )
        is_valid, new_metrics = evaluate(gt_labels, pred_labels)
        #TODO # put back or delete
        #print(new_metrics)
        #plot_four(image, mask.squeeze(), inner_mask, pred_labels)
        if not is_valid:
            continue
        else:
            accepted_validation_images += 1
        for metric_name, metric_val in new_metrics.items():
            metrics_dict[metric_name].append(metric_val)
    mean_metrics = {keym: np.mean(valuesm) for keym,valuesm in metrics_dict.items()}
    std_metrics = {keys: np.std(valuess) for keys,valuess in metrics_dict.items()}
    return {'mean': mean_metrics, 'std': std_metrics}

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