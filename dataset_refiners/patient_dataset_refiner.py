# Regular modules
import os
import os.path as pth
import numpy as np
import h5py
import json
from tqdm import tqdm
from time import time
import pandas as pd
import re
import cv2
from collections import defaultdict
import imagesize

# Custom modules
from image_slices_viewer import display_3d_image
import transforms as CT

storage_batch_name = "_storage_batch"
verbose = True

def create_meta_file(batch_size, partitions, partition_sizes,
                     counts, dataset_name, dimensions, labels):
    # Create file that describe the new refined dataset
    train_cases = partition_sizes['train']
    val_cases = partition_sizes['val']
    total_cases = train_cases + val_cases
    train_remainder = train_cases % batch_size
    train_batches = train_cases // batch_size + bool(train_remainder)
    val_remainder = val_cases % batch_size
    val_batches = val_cases // batch_size + bool(val_remainder)
    total_batches = train_batches + val_batches
    meta_path = pth.join(dataset_name, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'batch_size': batch_size,
            'remainder_training_batch_size': train_remainder,
            'remainder_validation_batch_size': val_remainder,
            'number_of_training_batches': train_batches,
            'number_of_validation_batches': val_batches,
            'total_number_of_batches': total_batches,
            'number_of_training_cases': train_cases,
            'number_of_validation_cases': val_cases,
            'total_number_of_cases': total_cases,
            'training_ratio': train_cases / total_cases,
            'validation_ratio': val_cases / total_cases,
            'number_of_cases_per_label': counts,
            'labels': dict(labels),
            'volume_dimensions': dimensions,
            'volume_dimensions_description': "(channels, depth, height, width)",
            'key_to_access_inputs': "data",
            'key_to_access_labels': "labels",
            'patients_in_partition': partitions
        }, f, indent=4)

def create_storage_batch(data, labels, dimensions, batch_number, dataset_name, folder):
    # Making a hdf5 file with data
    if verbose: tqdm.write("[%d] Saving batch..." % batch_number)
    start = time()
    name = str(batch_number) + storage_batch_name + ".h5"
    partition_path = pth.join(dataset_name, folder)
    path = pth.join(partition_path, name)
    if not pth.isdir(partition_path):
        os.makedirs(partition_path)
    f = h5py.File(path, "w")
    dset = f.create_dataset("data", (len(data), *dimensions), dtype='float32', compression="gzip")
    lset = f.create_dataset("labels", (len(data),), dtype='int8', compression="gzip")
    dset[()] = data
    lset[()] = labels
    elapsed = time() - start
    m, s = elapsed // 60, elapsed % 60
    if verbose: tqdm.write("    Batch has been saved! {%dm:%.1fs}" % (m, s))

def get_size(source_dir):
    return sum([len(dir_names) for _, dir_names, _ in os.walk(source_dir)])

def store(storage, storage_labels, shape, batch_number, new_dataset_name, partition_name):
    storage_array = np.stack(storage, axis=0)
    label_array = np.stack(storage_labels, axis=0)
    create_storage_batch(storage_array, label_array, shape, batch_number, new_dataset_name, partition_name)
    storage.clear()
    storage_labels.clear()

def to_float(image):
    min_val = np.min(image.ravel())
    max_val = np.max(image.ravel())
    out = (image.astype('float') - min_val) / (max_val - min_val)
    return out

def get_volume(case_path):
    files = [image for image in os.listdir(case_path) if image.endswith('.tif')]
    volume = []
    for file_name in files:
        image_path = pth.join(case_path, file_name)
        img = to_float(cv2.imread(image_path, 0))
        volume.append(img)
    volume = [volume[i] for i in range(len(volume) - 1)
                        if volume[i - 1].shape == volume[i].shape
                       and volume[i].shape == volume[i + 1].shape]
    return np.expand_dims(np.stack(volume).astype('float32'), axis=0)

def generate(source_dir, partitions, patient_paths, patient_labels, categories,
              counts, storage_batch_size=1, volume_transform=None, view_mode=False):
    new_dataset_name = source_dir + "_refined"
    partition_sizes = defaultdict(int)
    batch_number = 0
    shape = None
    pbar = tqdm(desc="Storing", total = get_size(source_dir), ncols=100)
    for partition_name, partition in partitions.items(): # For training and validation
        batch_data = []
        batch_labels = []
        for patient_id in partition: # For patients in partition
            for path in patient_paths[patient_id]: # For each scan belonging to patient
                volume = get_volume(path)
                if volume_transform:
                    volume = volume_transform(volume)
                if view_mode:
                    display_3d_image(volume)
                    continue
                if not shape:
                    shape = volume.shape
                
                batch_data.append(volume)
                batch_labels.append(patient_labels[patient_id])
                partition_sizes[partition_name] += 1
                pbar.update(1)
                if len(batch_data) >= storage_batch_size:
                    store(batch_data, batch_labels, shape, batch_number,
                          new_dataset_name, partition_name)
                    batch_number += 1
        if batch_data:
            store(batch_data, batch_labels, shape, batch_number,
                  new_dataset_name, partition_name)
            batch_number += 1
    
    create_meta_file(storage_batch_size, partitions, partition_sizes,
                     counts, new_dataset_name, shape, categories)
    pbar.close()

def get_patient_size(group):
    if not group:
        return 0
    return sum([patient[1] for patient in group])

def remove(patient, groups):
    for group in groups:
        if patient in group:
            group.remove(patient)

def strip_sizes(groups):
    groups = list(groups)
    for i, group in enumerate(groups):
        groups[i] = [patient_id for patient_id, size in group]
    return groups

def division_in_ratio(train_ratio, patients):
    '''This algorithm attempts to split a list of atomic units of
    various sizes into two groups of equal total sizes. A counter weight
    is used in intialization to shift the result from being an
    equal split to one fitting the given ratio.
    
    Always pick the biggest unit and place it in the smallest bin,
    for each iteration. The bin being min_group is swapped around so
    it is always the smalles bin holding the title of min_group'''
    total_size = get_patient_size(patients)
    diff = 2 * np.ceil(total_size * train_ratio) - total_size
    counter_weight = (-1, diff)
    groups = [list() for _ in range(2)]
    groups[0].append(counter_weight)
    for x in sorted(patients, reverse=True, key=lambda x: x[1]):
        min_group = groups[int(diff > 0)]
        for group in groups:
            if get_patient_size(group) < get_patient_size(min_group):
                min_group = group
        min_group.append(x)
    remove(counter_weight, groups)
    groups = sorted(groups, key=lambda x: get_patient_size(x), reverse=train_ratio > 0.5)
    groups_stripped = strip_sizes(groups)
    groups_dict = {"train": groups_stripped[0], "val": groups_stripped[1]}
    return groups_dict, get_patient_size(groups[0]) / total_size

def get_patient_sizes(patient_paths):
    return [(patient_id, len(paths)) for patient_id, paths in patient_paths.items()]

def is_valid_volume(case_path, dimensions):
    dimensions = np.array(dimensions)
    files = [image for image in os.listdir(case_path) if image.endswith('.tif')]
    volume = []
    for file_name in files:
        image_path = pth.join(case_path, file_name)
        image_shape = np.array(imagesize.get(image_path))[::-1]
        if (np.array(image_shape) >= dimensions[-2]).all():
            volume.append(image_shape.tolist())
        elif verbose:
            tqdm.write("- Skipping image at '%s', due to wrong dimensions!" % image_path)
    volume = [volume[i] for i in range(len(volume) - 1)
                        if volume[i - 1] == volume[i] and
                           volume[i] == volume[i + 1]]
    if len(volume) < dimensions[1]:
        if verbose:
            tqdm.write("- Skipping volume at '%s', due to wrong depth!" % case_path)
        return False
    return True

def filter_overview(source_dir, dimensions, patient_overview, volume_transform=None):
    patient_paths = defaultdict(lambda: defaultdict(list))
    patient_labels = dict()
    count = defaultdict(int)
    shift = 0
    max_id = -1
    categories = [c for c in os.listdir(source_dir) if pth.isdir(pth.join(source_dir, c))]
    categories = list(enumerate(sorted(categories)))
    pbar = tqdm(desc="Filtering", total = get_size(source_dir), ncols=100)
    for i, category in categories:
        category_path = pth.join(source_dir, category)
        for case in os.listdir(category_path):
            file_path = pth.join(category_path, case)
            if is_valid_volume(file_path, dimensions):
                patient_id = patient_overview[category][int(case)] + shift
                max_id = max(max_id, patient_id)
                count[category] += 1
                patient_paths[category][patient_id].append(file_path)
                patient_labels[patient_id] = i
            pbar.update(1)
        shift += max_id
    pbar.close()
    return patient_paths, patient_labels, categories, count

def get_patient_overview(path, num_control_cases=46, control_patient_size=4):
    # num_control_cases and control_patient_size must be set due to insufficient patients.xlsx
    amd = pd.read_excel(path, index_col='Unnamed: 0').iloc[1:, :2]
    result = {"amd": dict(), "control": dict()}
    for i, row in enumerate(np.array(amd)):
        ids = []
        info_string = ' '.join((str(val) for val in row))
        info_string = re.sub(r'[^\d -]', '', info_string) # Remove everything but numbers, space and '-'
        for match in re.findall(r'\d+-\d+', info_string): # Find matches for number ranges i.e. 5-10
            indices = re.findall(r'\d+', match)
            ids += list(range(int(indices[0]), int(indices[1]) + 1)) # Make the range into a list of numbers
        info_string = re.sub(r'\d+-\d+', '', info_string) # Remove number ranges
        ids += [int(match) for match in re.findall(r'\d+', info_string)] # Add any remaining single numbers
        for volume_id in ids:
            result["amd"][volume_id] = i # Link each volume id to a patient id
    for i in range(num_control_cases):
        result["control"][i + 1] = int(np.ceil(i / control_patient_size)) + 1
    return result

def partition_data(patient_paths, train_ratio):
    patient_sizes_amd = get_patient_sizes(patient_paths['amd'])
    patient_sizes_control = get_patient_sizes(patient_paths['control'])
    partitions, _ = division_in_ratio(train_ratio, patient_sizes_amd)
    partitions_control, _ = division_in_ratio(train_ratio, patient_sizes_control)
    partitions['train'] += partitions_control['train']
    partitions['val'] += partitions_control['val']
    patient_paths = {**patient_paths['amd'], **patient_paths['control']}
    return partitions, patient_paths

def prepare_partition(source_path, patients_path, dimensions, train_ratio=0.7, volume_transform=None):
    overview = get_patient_overview(patients_path)
    info = filter_overview(source_path, dimensions, overview, volume_transform)
    patient_paths, patient_labels, categories, counts = info
    partitions, patient_paths = partition_data(patient_paths, train_ratio)
    return partitions, patient_paths, patient_labels, categories, counts

if __name__ == '__main__':
    # Shape of one case is (channels, depth, height, width) in accourdance to how pytorch do convolution
    # We only have one channel in our volumes, thus (1, depth, height, width)
    cropped_size = (1, 32, 384, 384)
    slice_shape = (256, 256)
    v_transform = CT.Compose([
        CT.Crop(*cropped_size[1:]),
        CT.ResizeSlices(*slice_shape),
        CT.NormalizeBySlice()
    ])
    prep_args = prepare_partition(pth.join("..", "datasets", "st_olavs"),
                                  pth.join("..", "datasets", "st_olavs", "patients.xlsx"),
                                  cropped_size,
                                  train_ratio=0.7,
                                  volume_transform=v_transform)
    generate(pth.join("..", "datasets", "st_olavs"),
             *prep_args,
             storage_batch_size=24,
             volume_transform=v_transform,
             view_mode=False)