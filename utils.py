import numpy as np
import torch
from torch.cuda import current_device, get_device_capability, is_available
import os
import model
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

_COMPATIBLE = None

def set_compatibility(compatibility):
    global _COMPATIBLE
    _COMPATIBLE = compatibility

def is_compatible():
    global _COMPATIBLE
    if _COMPATIBLE is None:
        capability = get_device_capability(current_device())
        major = capability[0]
        _COMPATIBLE = not (capability == (3, 0) or major < 3)
    return _COMPATIBLE

def to_cpu(elements):
    if type(elements) == tuple or type(elements) == list:
        return [x.cpu() for x in elements]
    return elements.cpu()

def to_cuda(elements):
    if is_available() and is_compatible():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements

def init_weights(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, model.Conv2_1d):
        for mm in (m.conv2d, m.conv1d):
            torch.nn.init.xavier_normal_(mm.weight)
            if mm.bias is not None:
                mm.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.Linear) and 0:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def one_hits(predictions, targets):
    mask = targets == 1
    pred_masked = predictions[mask]
    if len(pred_masked) == 0:
        total = 0
    else:
        total = pred_masked.sum().item()
    return (total + 1) / (targets.sum().item() + 1)

def get_stats(predictions, targets):
    pt_mask = targets == 1
    nt_mask = targets == 0
    np_mask = predictions == 0

    tp = predictions[pt_mask].sum().item()
    fp = predictions[nt_mask].sum().item()
    fn = targets[np_mask].sum().item()
    tn = (1 - predictions[nt_mask]).sum().item()
    return tp, fp, fn, tn

def store_stats(val_loss, val_accuracy, precision, recall, amd_ratio, step, writer, stat_dict):
    # For live monitoring
    writer.add_scalars('Loss', {"Validation": val_loss}, global_step=step)
    writer.add_scalars('Accuracy', {"Validation": val_accuracy}, global_step=step)
    writer.add_scalars('Stats', {"Precision": precision}, global_step=step)
    writer.add_scalars('Stats', {"Recall": recall}, global_step=step)
    writer.add_scalar('AMD Ratio', amd_ratio, global_step=step)

    # For figures and documentation
    if stat_dict is not None:
        stat_dict['validation_loss'].append((step, val_loss))
        stat_dict['validation_accuracy'].append((step, val_accuracy))
        stat_dict['precision'].append((step, precision))
        stat_dict['recall'].append((step, recall))
        stat_dict['amd_ratio'].append((step, amd_ratio))


def validate(model, criterion, loader, step, writer, stat_dict=None):
    with torch.no_grad():
        model = model.eval()
        loss, accuracy, ratio, tp, fn, fp = 0, 0, 0, 0, 0, 0
        batch_size = len(loader)
        for batch in loader:
            inputs, targets = to_cuda(batch)

            outputs = model(inputs)
            predictions = torch.round(outputs)
            
            accuracy += (predictions == targets).sum().item() / targets.size(0)
            loss += criterion(outputs, targets)
            ratio += predictions.sum() / len(predictions)
            s_tp, s_fp, s_fn, _ = get_stats(predictions, targets)
            tp += s_tp
            fp += s_fp
            fn += s_fn
        
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        store_stats(
            val_loss=loss.item() / batch_size,
            val_accuracy=accuracy / batch_size,
            precision=prec,
            recall=rec,
            amd_ratio=ratio / batch_size,
            step=step,
            writer=writer,
            stat_dict=stat_dict
        )

        model = model.train()
    return accuracy

def save_model(model, optimizer, folder_path, name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    path = os.path.join(folder_path, name)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, path)

def load_model(model, optimizer, path):
    pre_trained_dict = torch.load(path)

    # Union of dictionaries where pre-trained_dict overwrites values for overlapping keys
    # Loading the fused dictionary into model and optimizer
    if model is not None:
        model_dict = model.state_dict()
        fused_model = dict(model_dict, **pre_trained_dict['model_state_dict'])
        model.load_state_dict(fused_model)
    if optimizer is not None:
        optim_dict = optimizer.state_dict()
        fused_optim = dict(optim_dict, **pre_trained_dict['optimizer_state_dict'])
        optimizer.load_state_dict(fused_optim)
    return model, optimizer

def print_model_info(path):
    # Get info on model
    path = os.path.join(path, 'hyper_params.pth')
    info = torch.load(path)
    for key, value in info.items():
        print(key + ':', value)

def calculate_roc_stats(total_targets, total_outputs, stats_path=None, show_roc=False):
    auc_score = roc_auc_score(total_targets, total_outputs)

    if show_roc or stats_path:
        # BE AWARE!!! Positive (AMD) label is assumed to be 1! This is also defined in dataset.
        fpr, tpr, thresholds = roc_curve(total_targets, total_outputs, pos_label=1)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        if stats_path:
            plt.savefig(os.path.join(stats_path, 'roc.png'))
        if show_roc:
            plt.show()
    return auc_score

class CorrectShape:
    def __call__(self, batch):
        data, labels = zip(*batch)
        data = torch.stack(data)
        labels = torch.cat(labels)
        return data, labels