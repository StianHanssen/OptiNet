# Regular modules
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# PyTorch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Custom modules
import model as m
from utils import to_cpu, to_cuda, is_compatible, load_model, print_model_info, get_stats, set_compatibility
from dataset import AMDDataset
import fixations as fx
import fxutils

def calculate_fixations(preditions, model, use_2_1d_conv):
    fixations_conv3d = fx.fixations_conv2_1d if use_2_1d_conv else fx.fixations_conv
    pbar = tqdm(desc="Fixations progress in steps", total = 16, ncols=100)
    preditions = to_cpu(preditions)
    model = to_cpu(model)
    x = fx.fixations_fc(preditions, model.classifier[2])
    pbar.update(1)
    x = fx.fixations_fc(x, model.classifier[0])
    pbar.update(1)
    x = fx.fixations_maxpool(x, model.feature[6]) # down_sample2
    pbar.update(1)
    x = fixations_conv3d(x, model.feature[4]) # conv3x3x3
    pbar.update(1)
    for i in range(3, -1, -1): # Iterate over the blocks in model.feature
        x = fx.fixations_maxpool(x, model.feature[i].net[-1]) # Block.down_sample
        pbar.update(1)
        x = fixations_conv3d(x, model.feature[i].conv2) # Block.conv2
        pbar.update(1)
        x = fixations_conv3d(x, model.feature[i].conv1) # Block.conv1
        pbar.update(1)
    pbar.close()
    model = to_cuda(model)
    return x

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()
    print("Cuda available:", torch.cuda.is_available() and is_compatible())

    # Setup paramameters
    name = 'pakistan'
    version = 'best'
    val_batch_size = 1
    print_model = False
    test_dataset = 'st_olavs_refined'
    use_2_1d_conv = True
    visualize = False
    store_data = True

    conv3d_module = m.Conv2_1d if use_2_1d_conv else nn.Conv3d

    # Paths
    model_path = os.path.join('saved_models', test_dataset, name, 'AMDModel_%s.pth' % version)
    validation_path = os.path.join('datasets', test_dataset, 'val')
    fixations_path = os.path.join('fixations', test_dataset, name)

    # Creating folder for current run
    if not os.path.exists(fixations_path) and store_data:
        os.makedirs(fixations_path)

    # Dataset and data loader
    validation_dataset = AMDDataset(validation_path, one_hot=False, use_transforms=False)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=val_batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   drop_last=False)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()

    #Print model info
    if print_model:
        print_model_info(name)

    # Model
    model = to_cuda(m.AMDModel(conv3d_module,
                                   stride=1,
                                   bias=True))
    model, _ = load_model(model, None, model_path)
    model.eval()

    # Evaluation
    storage_dict = defaultdict(list)
    data_size = len(validation_loader)
    with torch.no_grad():
        for batch in validation_loader:
            inputs, targets = to_cuda(batch)
            with fx.record_activations(model):
                print("Predicting...")
                outputs = model(inputs)
            predictions = torch.round(outputs)
            batch_num = torch.LongTensor([[i for i in range(len(predictions))]]).t()
            pred = torch.cat([batch_num, torch.zeros(predictions.shape).long()], dim=1)
            points = calculate_fixations(pred, model, use_2_1d_conv).numpy()
            points = fxutils.chunk_batch(points)
            if visualize:
                for i in range(len(inputs)):
                    fxutils.visualize(inputs[i], points[i], diag_percent=0.1, image_label=targets[i], prediction=predictions[i])
            if store_data:
                storage_dict['inputs'].append(inputs)
                storage_dict['outputs'].append(outputs)
                storage_dict['targets'].append(targets)
                storage_dict['points'] += points
    if store_data:
        storage_dict['inputs'] = torch.cat(storage_dict['inputs'], dim=0)
        storage_dict['outputs'] = torch.cat(storage_dict['outputs'], dim=0)
        storage_dict['targets'] = torch.cat(storage_dict['targets'], dim=0)
        torch.save(storage_dict, os.path.join(fixations_path, "fixation_data.pth"))
    