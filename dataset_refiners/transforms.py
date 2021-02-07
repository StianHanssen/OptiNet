# Regluar modules
from random import random
import math
from functools import wraps
import time
import cv2

def my_timer(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print()
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper

import time


def Compose(transforms):
    def output(x):
        for transform in transforms:
            x = transform(x)
        return x
    return output

def ResizeSlices(t_height, t_width):
    def resize_slices(tensor):
        new_tensor = tensor[:, :, :t_height, :t_width]
        for i, frame in enumerate(tensor[0]):
            new_tensor[0, i] = cv2.resize(frame, (t_height, t_width))
        return new_tensor
    return resize_slices

def RandomHorizontalFlip(probability):
    def output(tensor):
        if random() < probability:
            tensor = tensor.flip(3)
        return tensor
    return output

def NormalizeBySlice():
    def normalize_by_slice(tensor):
        for i, frame in enumerate(tensor[0]):
            stdev = frame.std()
            mean = frame.mean()
            tensor[0, i] = (frame - mean) / stdev
        return tensor
    return normalize_by_slice

def Normalize():
    def normalize(tensor):
        stdev = tensor.std()
        mean = tensor.mean()
        tensor = (tensor - mean) / stdev
        return tensor
    return normalize

def get_crop_borders(value, target, offset=0):
    '''
    Gives idices for center cropping, a decimal number can be given for
    offset to shift the crop. 
    Example: If we crop in width with an offset of 0.5 where
    the number of pixels cropped is 10 on right and left side. The offset
    will move 0.5 of the cropping from left side to right side. Thus
    you will crop 5 pixels on left side and 15 pixels and right side.
    '''
    if value > target:
        diff = (value - target) / 2
        start = math.floor(diff)
        end = math.ceil(diff)
        shift = math.floor(start * offset)
        start -= shift
        end += shift
        return start, -end
    return 0, value

def Crop(t_depth, t_height, t_width, offset=0):
    '''
    Assumes volume of shape (C, D, H, W), however the crop will only work on (D, H, W). 
    Offset can be set to 1 if you want the same offset for all dimensions or
    with a tuple of shape (D, H, W).
    '''
    if isinstance(offset, int):
        offset = (offset, offset, offset)
    depth_offset, height_offset, width_offset = offset
    # Crops volume in 3 dimensions
    def crop(volume):
        _, depth, height, width = volume.shape
        d_start, d_end = get_crop_borders(depth, t_depth, depth_offset)
        w_start, w_end = get_crop_borders(width, t_width, width_offset)
        h_start, h_end = get_crop_borders(height, t_height, height_offset)
        return volume[:, d_start:d_end, h_start:h_end, w_start:w_end]
    return crop