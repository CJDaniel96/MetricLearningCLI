import os
import re
import numpy as np
import torch
import random
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import DOLGModel, EfficientArcFaceModel


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
            transforms.ToTensor()
        ])
    elif mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

def get_mean_std(data_dir: Path, batch_size):
    image_datasets = ImageFolder(str(data_dir.joinpath('train')), select_data_transforms())
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

def save_mean_std(data_dir: Path, mean, std):
    with data_dir.joinpath('mean_std.txt').open('w') as f:
        f.write(f'mean: {mean}\n')
        f.write(f'std: {std}')

def read_mean_std(mean_std_file: Path):
    if mean_std_file.exists():
        with mean_std_file.open('r') as f:
            mean = eval(re.search('[\[].*[]]', f.readline())[0])
            std = eval(re.search('[\[].*[]]', f.readline())[0])
    else:
        raise ValueError('Mean and std file not found')

    return mean, std

def save_class_to_idx(data_dir: Path, class_to_idx):
    with data_dir.joinpath('class_to_idx.txt').open('w') as f:
        for classes in class_to_idx:
            f.write(f'{classes} {class_to_idx[classes]}\n')
            

class EarlyStopping:
    """
    Early stopping class for monitoring the validation loss during training.
    
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped. Default is 5.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
    
    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        best_loss (float): Best validation loss achieved so far.
        counter (int): Number of epochs with no improvement.
        early_stop (bool): Flag indicating whether to stop training or not.
    """
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Update the best validation loss and check if training should be stopped.
        
        Args:
            val_loss (float): Current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                

class ModelFactory:
    """
    A factory class for creating models.
    """

    @staticmethod
    def create_model(model_structure, model_path, embedding_size):
        """
        Creates a model based on the given model structure.

        Args:
            model_structure (str): The model structure to create.
            model_path (str): The path to the model weights.
            embedding_size (int): The size of the embedding.

        Returns:
            model: The created model.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_structure == 'EfficientArcFaceModel':
            model = EfficientArcFaceModel(embedding_size=embedding_size).to(device)
        elif model_structure == 'DOLG':
            model = DOLGModel(embedding_size=embedding_size).to(device)
        else:
            raise ValueError('Invalid model structure')
        
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        
        return model


def load_model(model_structure, model_path, embedding_size):
    """
    Load a pre-trained model.

    Args:
        model_structure (str): The structure of the model to be loaded.
        model_path (str): The path to the pre-trained model file.
        embedding_size (int): The size of the embedding layer.

    Returns:
        model: The loaded pre-trained model.

    """
    model = ModelFactory.create_model(model_structure, model_path, embedding_size)
    return model

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(path, dataset_mode='default', imgsz=224, batch_size=8, shuffle=True, num_workers=8, seed=42, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], drop_last=False):
    transform = select_data_transforms(dataset_mode, mean, std, imgsz)
    dataset = ImageFolder(path, transform)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, num_workers])  # number of workers
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nw, pin_memory=True, worker_init_fn=seed_worker, generator=generator, drop_last=drop_last), dataset
    

class UnNormalize(object):
    """
    A class to unnormalize a tensor using mean and standard deviation.
    
    Args:
        mean (list or tuple): The mean values for each channel.
        std (list or tuple): The standard deviation values for each channel.
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        """
        Unnormalize the given tensor.
        
        Args:
            tensor (torch.Tensor): The tensor to be unnormalized.
        
        Returns:
            torch.Tensor: The unnormalized tensor.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor