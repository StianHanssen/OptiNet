'''
Credits: https://github.com/val-iisc/cnn-fixations/blob/master/demo/utils.py
The following functions are taken from the original implementation:
 - outlier_removal
 - heatmap (small modification allowing to set k, adaptable sigma)
 - visualize (small modification allowing to set k, red dots)
'''
import numpy as np
import torch
from torch.cuda import current_device, get_device_capability, is_available
import cv2
import scipy.ndimage.filters as scifil
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from numpy import ravel_multi_index, unravel_index, unique as npunique
import oct_result_viewer as orv

def as_tensor(iterable):
    return torch.LongTensor(list(iterable))

def slice_weight(lower_bound, upper_bound, shape, padding, dilation, act_shape): # UNFINISHED
    act_shape += padding
    lower_crop = -(lower_bound - padding) / dilation
    lower_crop = torch.where(lower_crop > 0, lower_crop, torch.zeros(lower_crop.shape, dtype=torch.long))
    upper_crop = (act_shape - upper_bound) / dilation
    upper_crop = torch.where(upper_crop > 0, upper_crop, shape)

def get_slicer(neuron_pos, kernel_size, stride, dilation):
    lower_bound = neuron_pos * stride
    lower_bound[0] = 0
    size = kernel_size + (kernel_size - 1) * (dilation - 1)
    upper_bound = lower_bound + size
    slicer = tuple(slice(lower_bound[i].item(), upper_bound[i].item(), dilation[i].item()) for i in range(len(upper_bound)))
    return slicer, lower_bound

def unflatten(indices, shape):
    return torch.LongTensor(unravel_index(indices, list(shape))).t()

def flatten(points, shape):
    return torch.from_numpy(ravel_multi_index(points.t().numpy(), list(shape)))

def convert_flat_fixations(fixations, layer_info):
    if len(fixations[0]) == 2:
        shape = layer_info.get_out_shape()
        expanded = unflatten(fixations[:, 1], shape[1:])
        fixations = torch.cat([fixations[:, 0].unsqueeze(1), expanded], dim=1)
    return fixations

def unique(fixations, d=False):
    return torch.from_numpy(npunique(fixations.numpy(), axis=0))

def chunk_batch(points, remove_chanels=True):
    '''
    Points originally has the format [fixation1, fixations2..], where
    fixation[0] = mini batch index. Chunk batch transform  points format:
    [[fixation1, fixation2, ...], [fixation53, fixations54..], ...] so that
    points[0] gives you all fixations belonging to mini batch 0.
    '''
    if remove_chanels:
        points = np.concatenate([points[:, :1], points[:, 2:]], axis=1)
    ordered = points[points[:, 0].argsort(axis=0)] # Order so points from a batch follow eachother
    batch_sizes = np.bincount(ordered[:, 0]).tolist()
    section_indices = np.cumsum(batch_sizes)[:-1] # Get the divider between each mini batch
    return np.split(ordered[:, 1:], section_indices, axis=0)

def split_depth(points, depth_size):
    chunks = [[] for _ in range(depth_size)]
    for i in range(depth_size):
        chunks[i] = points[points[:, 0] == i][:, 1:]
    return chunks

def outlier_removal(points, diag):
    neighbors = np.zeros((points.shape[0]))
    selPoints = np.empty((1, *points[0].shape))
    for i in range(points.shape[0]):
        diff = np.sqrt(np.sum(np.square(points-points[i]), axis=1))
        neighbors[i] = np.sum(diff < diag)
    for i in range(points.shape[0]):
        if neighbors[i] > 0.05*points.shape[0]:
            selPoints = np.append(selPoints, points[i:i+1, :], axis=0)
    selPoints = selPoints[1:, :]
    selPoints = selPoints.astype(int)
    return selPoints

def heatmap(img, points, sigma=None, k=None, cut_channel=False):
    ind = -1 if cut_channel else len(img.shape)
    if k is None:
        k = (np.min(img.shape[:ind])) if (
            np.min(img.shape[:ind]) % 2 == 1) else (np.min(img.shape[:ind])-1)
    if sigma is None:
        sigma= 10 #k / 10.45
    mask = np.zeros(img.shape[:ind])
    shape = mask.shape
    # Add one to all positions that are within shape
    # Note: If overlapping points are added then we need to change this
    mask[tuple(points[np.all(points < shape, axis=1)].T)] += 1
    # Gaussian blur the points to get a nice heatmap
    blur = scifil.gaussian_filter(mask, sigma)
    #blur = cv2.GaussianBlur(mask, (k, k), sigma)
    blur = blur*255/np.max(blur)
    return blur

def visualize(img, points, diag_percent, image_label, prediction, k=None):
    img, image_label, prediction = img.cpu(), image_label.cpu(), prediction.cpu()
    shape = img.shape
    cut_channel = True
    if (len(shape) == 3 or len(shape) == 4) and shape[0] == 1:
        img.squeeze_(0)
        cut_channel = False
    elif len(shape) == 3 and shape[0] == 3:
        img = img.permute(1, 2, 0).contiguous()
    diag = np.sqrt(sum([dim**2 for dim in img.shape]))*diag_percent
    selPoints = outlier_removal(points, diag)
    #selPoints = points
    # Make heatmap and show images
    hm = heatmap(np.copy(img), selPoints, k=k, cut_channel=cut_channel)
    if len(shape) == 4 and shape[0] == 1:
        selPoints = split_depth(selPoints, shape[1])
    if len(shape) == 3:
        display2d(img, selPoints, hm, image_label)
    elif len(shape) == 4:
        orv.display_3d_image(img, selPoints, hm, image_label, prediction)

def display2d(img, points, heatmap, image_label):
    vmin, vmax = img.min(), img.max()
    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax), ax[0].axis('off'), ax[0].set_title(image_label)
    ax[1].imshow(img, cmap='gray', vmin=vmin, vmax=vmax), ax[1].axis('off'),
    ax[1].scatter(points[:, 1], points[:, 0], c='r'),
    ax[1].set_title('CNN Fixations')
    ax[2].imshow(img, cmap='gray', vmin=vmin, vmax=vmax), ax[2].imshow(heatmap, 'jet', alpha=0.6)
    ax[2].axis('off'), ax[2].set_title('Heatmap')
    plt.show()

def show_saved_fixations(path):
    fixations_dict = torch.load(path)
    inputs = fixations_dict['inputs']
    points = fixations_dict['points']
    outputs = fixations_dict['outputs']
    targets = fixations_dict['targets']
    predictions = torch.round(outputs)
    for i in range(len(inputs)):
        visualize(inputs[i], points[i], diag_percent=0.1, image_label=targets[i], prediction=predictions[i])