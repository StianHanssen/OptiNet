# Regular modules
import os
from os.path import join, isdir
import numpy as np
from skimage import io
import h5py
import json
from tqdm import tqdm
import math
from time import time
from scipy import ndimage
import pandas as pd
import re

# Custom modules
from image_slices_viewer import display_3d_image
import transforms as CT

storage_batch_name = "_storage_batch"
verbose = True

def get_patient_overview(path, control_size=24, control_patient_size=4):
    amd = pd.read_excel(path, index_col=None).iloc[1:,:3]
    result = {"amd": dict(), "control": dict()}
    for i, patient in enumerate(amd.iloc[:, 0]):
        ids = []
        info_string = str(amd.iloc[i, 1]) + " " + str(amd.iloc[i, 2])
        info_string = re.sub(r'[^\d -]', '', info_string) # Remove everything but numbers, space and '-'
        for match in re.findall(r'\d+-\d+', info_string): # Find matches for number ranges i.e. 5-10
            indices = re.findall(r'\d+', match)
            ids += list(range(int(indices[0]), int(indices[1]))) # Make the range into a list of numbers
        info_string = re.sub(r'\d+-\d+', '', info_string) # Remove number ranges
        ids += [int(match) for match in re.findall(r'\d+', info_string)] # Add any remaining single numbers
        for id in ids:
            result["amd"][id] = patient # Link each volume id to a patient id
    for i in range(1, control_size * control_patient_size + 1):
        result["control"][i] = math.ceil(i / control_patient_size)
    return result

def partition(classwise_data, train_ratio, ratio, shuffle=False):
    sizes = np.array([len(data) for data in classwise_data])
    ratios = sizes / sizes.sum()
    if shuffle:
        classwise_data = [np.random.shuffle(data) for data in classwise_data]
    split_indices = int(np.ceil(sizes * train_ratio))
    train_partition = [classwise_data[i][:splitter] for i, splitter in enumerate(split_indices)]
    val_partition = [classwise_data[i][splitter:] for i, splitter in enumerate(split_indices)]
    return train_partition, val_partition, ratios

def create_meta_file(batch_size, batch_number, last_batch_size, labels, dataset_name, dimensions):
    # Create file that describe the new refined dataset
    meta_path = join(dataset_name, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'batch_size': batch_size,
            'remainder_batch_size': last_batch_size,
            'number_of_batches': batch_number + (1 if last_batch_size else 0),
            'total_number_of_cases': batch_size * batch_number + last_batch_size,
            'labels': dict(labels),
            'volume_dimensions': dimensions,
            'volume_dimensions_description': "(channels, depth, height, width)",
            'key_to_access_inputs': "data",
            'key_to_access_labels': "labels",
        }, f, indent=4)

def create_storage_batch(data, labels, dimensions, batch_number, dataset_name):
    # Making a hdf5 file with data
    if verbose: tqdm.write("[%d] Saving batch..." % batch_number)
    start = time()
    name = str(batch_number) + storage_batch_name + ".h5"
    path = join(dataset_name, name)
    if not isdir(dataset_name):
        os.makedirs(dataset_name)
    f = h5py.File(path, "w")
    dset = f.create_dataset("data", (len(data), *dimensions), dtype='float32', compression="gzip")
    lset = f.create_dataset("labels", (len(data),), dtype='int8', compression="gzip")
    dset[()] = data
    lset[()] = labels
    elapsed = time() - start
    m, s = elapsed // 60, elapsed % 60
    if verbose: tqdm.write("    Batch has been saved! {%dm:%.1fs}" % (m, s))

def get_valid_volume(case_path):
    dimensions = None
    files = [image for image in os.listdir(case_path) if image.endswith('.tif')]
    volume = []
    for file_name in files:
        image_path = join(case_path, file_name)
        img = io.imread(image_path, as_gray=True)
        if dimensions is None: # Take first image as template dimensions for a slice
            dimensions = img.shape
        if img.shape == dimensions: # It is assumed all images have the same dimensions, avoid ones that do not fit
            volume.append(img)
        elif verbose:
            tqdm.write("- Skipping image at '%s', due to wrong dimensions!" % image_path)
        if not volume:
            return None
    return np.expand_dims(np.stack(volume).astype('float32'), axis=0)

def get_size(source_dir):
    return sum([len(dir_names) for _, dir_names, _ in os.walk(source_dir)])

def generate(source_dir, dimensions, storage_batch_size=1, volume_transform=None, view_mode=False):
    # Generate new refined dataset
    new_dataset_name = source_dir + "_refined"
    data = None if dimensions is None else np.ndarray((storage_batch_size, *dimensions), dtype='float32')
    labels = np.ndarray(storage_batch_size, dtype='float32')
    
    case_index = 0
    batch_number = 0

    categories = [c for c in os.listdir(source_dir) if isdir(join(source_dir, c))]
    categories = list(enumerate(sorted(categories)))
    pbar = tqdm(desc="Progress", total = get_size(source_dir), ncols=100)
    for i, category in categories:
        category_path = join(source_dir, category)
        for case in os.listdir(category_path):
            file_path = join(category_path, case)
            volume = get_valid_volume(file_path)
            pbar.update(1)
            if volume is None:
                if verbose: tqdm.write("- Skipping case at '%s', found no images!" % file_path)
                continue

            if data is None:
                dimensions = volume.shape
                data = np.ndarray((storage_batch_size, *dimensions), dtype='float32')

            if volume_transform is not None:
                volume = volume_transform(volume)
            
            if view_mode:
                display_3d_image(volume)
                continue
            
            if volume.shape != dimensions:
                if verbose: 
                    tqdm.write("- Skipping case at '%s', due to wrong dimensions!" % file_path)
                    tqdm.write("  Volume shape: " + str(volume.shape))
                    tqdm.write("  Target shape: " + str(dimensions))
                continue

            data[case_index] = volume
            labels[case_index] = i
            case_index += 1

            if case_index == storage_batch_size:
                create_storage_batch(data, labels, dimensions, batch_number, new_dataset_name)
                batch_number += 1
                case_index = 0

    if case_index > 0:
        data = data[:case_index]
        labels = labels[:case_index]
        create_storage_batch(data, labels, dimensions, batch_number, new_dataset_name)
    create_meta_file(storage_batch_size, batch_number, case_index, categories, new_dataset_name, dimensions)
    pbar.close()

if __name__ == '__main__':
    # Shape of one case is (channels, depth, height, width) in accourdance to how pytorch do convolution
    # We only have one channel in our volumes, thus (1, depth, height, width)
    dimensions = (1, 32, 384, 384)
    scale = np.array((1, 1, 2/3, 2/3))
    v_transform = CT.Compose([
        CT.Crop(*dimensions, (0, 0, 0)),
        lambda x: ndimage.zoom(x, scale),
        CT.NormalizeBySlice()
    ])
    final_shape = tuple(int(dimensions[i] * scale[i]) for i in range(len(dimensions)))

    generate(source_dir=join("..", "datasets", "st_olavs"),
            dimensions=final_shape, 
            storage_batch_size=24,
            volume_transform=v_transform,
            view_mode=False)