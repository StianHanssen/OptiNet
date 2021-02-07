import os
import numpy as np
import h5py
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from dataset_refiners import transforms

class AMDDataset(Dataset):
    def __init__(self, root_dir, one_hot=False, use_transforms=True, return_ids=False):
        """
        Loads datasets of a certain structure (see refined datasets).
        AMD cases will always be defined by number 1, while control is 0.
        """
        self.root_dir = root_dir
        self.return_ids = return_ids
        #self.meta = self.load_meta()
        self.one_hot = one_hot
        self.use_transforms = use_transforms
        # NB! These are custom transforms to work with numpy, not PyTorch's
        self.transform = transforms.Compose([
            torch.Tensor,
            transforms.RandomHorizontalFlip(0.5)
        ])
        self.meta = self.load_meta()
        self.names = [fn for fn in os.listdir(root_dir) if fn.endswith(".h5")]
        self.data, self.labels, self.ids = self.load_data()
        if self.meta['labels']['0'] == 'amd':
            self.labels = 1 - self.labels # Set AMD to have label 1
        self.amd_ratio = self.labels.sum() / len(self.labels)
        print("AMD/Total ratio:", self.labels.sum(), "/", len(self.labels), "=", self.amd_ratio)
    
    def load_meta(self):
        with open(os.path.join(self.root_dir, '..', 'meta.json')) as json_data:
            return json.load(json_data)
    
    def load_batch(self, name):
        dataset = h5py.File(os.path.join(self.root_dir, name), mode='r')
        if self.return_ids:
            return dataset['data'][()], dataset['labels'][()], dataset['ids'][()]
        return dataset['data'][()], dataset['labels'][()]
    
    def load_data(self):
        data = []
        labels = []
        ids = []
        for name in tqdm(self.names, desc="Loading dataset", ncols=100):
            batch = self.load_batch(name)
            b_data, b_labels = batch[:2]
            data.append(b_data)
            labels.append(b_labels)
            if self.return_ids:
                ids.append(batch[2])
        
        data = np.concatenate(data).astype(np.float32)
        labels = np.concatenate(labels).astype(np.int8)
        if self.one_hot:
            one_hots = np.zeros((len(labels), labels.max() + 1))
            one_hots[np.arange(len(labels)), labels] = 1
            labels = one_hots
        if self.return_ids:
            ids = np.concatenate(ids).astype(np.int16)
            return data, labels, ids
        return data, labels, None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels[idx]
        if self.use_transforms:
            item = self.transform(item)
        else:
            item = torch.Tensor(item)
        if self.return_ids:
            case_id = self.ids[idx]
            return item, torch.FloatTensor([label]), torch.LongTensor([case_id])
        return item, torch.FloatTensor([label])

    @staticmethod
    def mix_datasets(dataset1, dataset2, indices=None):
        '''
        Mixes and shuffles the data from both datasets and distribute the data
        back into the datasets with the same length as they originally had.
        '''
        len1, len2 = len(dataset1.data), len(dataset2.data)
        if indices is None:
            indices = list(range(len1 + len2))
            np.random.shuffle(indices)
        all_data = np.concatenate([dataset1.data, dataset2.data], axis=0)
        all_labels = np.concatenate([dataset1.labels, dataset2.labels], axis=0)
        
        all_data = all_data[indices]
        all_labels = all_labels[indices]

        dataset1.data, dataset1.labels = all_data[:len1], all_labels[:len1]
        dataset2.data, dataset2.labels = all_data[len1:], all_labels[len1:]

        if dataset1.return_ids and dataset2.return_ids:
            all_ids = np.concatenate([dataset1.ids, dataset2.ids], axis=0)
            all_ids = all_ids[indices]
            dataset1.ids = all_ids[:len1]
            dataset2.ids = all_ids[len1:]

        assert len(dataset1.data) == len1, "Expected the dataset1 to have the same length after mix"
        assert len(dataset2.data) == len2, "Expected the dataset2 to have the same length after mix"
        return indices

