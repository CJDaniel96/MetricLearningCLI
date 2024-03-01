import os
import re
import numpy as np
import torch
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def setup_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def select_data_transforms(mode='default', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], image_size=224):
    assert mode in ['default', 'train'], 'mode is not validated'
    if mode == 'default':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            # transforms.CenterCrop((224, 224)), 
            transforms.ToTensor()
        ])
    elif mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            # transforms.CenterCrop((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

def get_mean_std(data_dir, batch_size):
    image_datasets = ImageFolder(os.path.join(data_dir, 'train'), select_data_transforms())
    dataloaders = DataLoader(image_datasets, batch_size=batch_size, shuffle=True)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloaders:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples

    return mean, std

def save_mean_std(data_dir, mean, std):
    with open(os.path.join(data_dir, 'mean_std.txt'), 'w') as f:
        f.write(f'mean: {mean}\n')
        f.write(f'std: {std}')

def read_mean_std(mean_std_file):
    with open(mean_std_file, 'r') as f:
        mean = eval(re.search('[\[].*[]]', f.readline())[0])
        std = eval(re.search('[\[].*[]]', f.readline())[0])

    return mean, std

def save_class_to_idx(data_dir, class_to_idx):
    with open(os.path.join(data_dir, 'class_to_idx.txt'), 'w') as f:
        for classes in class_to_idx:
            f.write(f'{classes} {class_to_idx[classes]}\n')