# Regular modules
import os
from os.path import join, isdir
import numpy as np
import h5py
import json
from scipy.io import loadmat
from torchvision import transforms
from tqdm import tqdm
from time import time
from scipy import ndimage

# Custom modules
from image_slices_viewer import display_3d_image
import transforms as CT

storage_batch_name = "_storage_batch"
verbose = True

def create_storage_batch(ids, data, labels, dimensions, batch_number, dataset_name):
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
    iset = f.create_dataset("ids", (len(data),), dtype='int16', compression="gzip")
    dset[()] = data
    lset[()] = labels
    iset[()] = ids
    elapsed = time() - start
    m, s = elapsed // 60, elapsed % 60
    if verbose: tqdm.write("    Batch has been saved! {%dm:%.1fs}" % (m, s))

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

def transform_dimensions(volume):
    # Rearranges dimensions to fit pytorch
    content_transform = transforms.Compose([transforms.ToTensor()])
    volume = content_transform(volume).numpy()
    volume = volume.reshape(1, *volume.shape)
    return volume

def get_size(source_dir):
    return sum([len(files) for _, _, files in os.walk(source_dir)])

def generate(source_dir, dimensions, storage_batch_size=1, volume_transform=None, view_mode=False):
    # Generate new refined dataset
    new_dataset_name = source_dir + "_refined"
    data = None if dimensions is None else np.ndarray((storage_batch_size, *dimensions), dtype='float32')
    labels = np.ndarray(storage_batch_size, dtype='int8')
    ids = np.ndarray(storage_batch_size, dtype='int16')

    batch_number = 0
    case_index = 0

    categories = [c for c in os.listdir(source_dir) if isdir(join(source_dir, c))]
    categories = list(enumerate(sorted(categories)))
    pbar = tqdm(desc="Progress", total = get_size(source_dir), ncols=100)
    for i, category in categories:
        category_path = join(source_dir, category)
        for case in os.listdir(category_path):
            file_path = join(category_path, case)
            mat_dict = loadmat(file_path)
            volume = transform_dimensions(mat_dict['images'])
            if data is None:
                dimensions = volume.shape
                data = np.ndarray((storage_batch_size, *dimensions), dtype='float32')

            if volume_transform is not None:
                volume = volume_transform(volume)
            
            pbar.update(1)
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
            ids[case_index] = int(case[case.rfind('_') + 1: -4])
            case_index += 1

            if case_index == storage_batch_size:
                create_storage_batch(ids, data, labels, dimensions, batch_number, new_dataset_name)
                batch_number += 1
                case_index = 0

    if case_index > 0:
        data = data[:case_index]
        labels = labels[:case_index]
        create_storage_batch(ids, data, labels, dimensions, batch_number, new_dataset_name)
    create_meta_file(storage_batch_size, batch_number, case_index, categories, new_dataset_name, dimensions)
    pbar.close()

if __name__ == '__main__':
    # Shape of one case is (channels, depth, height, width) in accourdance to how pytorch do convolution
    # We only have one channel in our volumes, thus (1, depth, height, width)
    dimensions = (1, 32, 512, 512)
    slice_shape = (256, 256)
    v_transform = CT.Compose([
        CT.Crop(*dimensions[1:], (0, 1, 0)),
        CT.ResizeSlices(*slice_shape),
        CT.NormalizeBySlice()
    ])
    final_shape = tuple([*dimensions[:2], *slice_shape])

    generate(source_dir=join("..", "datasets", "duke"),
            dimensions=final_shape, 
            storage_batch_size=24,
            volume_transform=v_transform,
            view_mode=False)