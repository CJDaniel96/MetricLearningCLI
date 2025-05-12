import os
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import json
from typing import List, Tuple, Dict, Optional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from pytorch_metric_learning import testers
from collections import defaultdict

from model import DOLGModel, EfficientArcFaceModel, MLGModel

# Constants
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_SEED = 42


def setup_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility across different libraries.

    Args:
        seed: The seed value to set (default: 42).
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class DataTransformFactory:
    """Factory class to create data transformation pipelines."""

    @staticmethod
    def create_transform(mode: str, mean: List[float] = DEFAULT_MEAN, std: List[float] = DEFAULT_STD,
                        image_size: int = 224) -> transforms.Compose:
        """Create a transformation pipeline based on the mode.

        Args:
            mode: The mode of transformation ('default', 'train', 'val', 'test').
            mean: Mean values for normalization (default: [0.485, 0.456, 0.406]).
            std: Standard deviation values for normalization (default: [0.229, 0.224, 0.225]).
            image_size: Target image size (default: 224).

        Returns:
            A composed transformation pipeline.

        Raises:
            AssertionError: If mode is not valid.
        """
        valid_modes = {'default', 'train', 'val', 'test'}
        if mode not in valid_modes:
            raise AssertionError(f"Mode must be one of {valid_modes}, got {mode}")

        base_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]

        if mode == 'train':
            base_transforms.append(transforms.Normalize(mean, std))

        return transforms.Compose(base_transforms)


class DataStatistics:
    """Class to compute and manage dataset statistics (mean and std)."""

    @staticmethod
    def compute_mean_std(dataloader: DataLoader) -> Tuple[Tensor, Tensor]:
        """Compute the mean and standard deviation of a dataset.

        Args:
            dataloader: DataLoader containing the dataset.

        Returns:
            A tuple of (mean, std) tensors.
        """
        mean, std = 0., 0.
        total_samples = 0.

        for data, _ in dataloader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            total_samples += batch_samples

        return mean / total_samples, std / total_samples

    @staticmethod
    def get_mean_std(data_dir: Path, batch_size: int = 8, mean_std_name: str = "mean_std.txt") -> Tuple[List[float], List[float]]:
        """Retrieve or compute dataset mean and standard deviation.

        Args:
            data_dir: Path to the dataset directory.
            batch_size: Batch size for computing statistics.

        Returns:
            A tuple of (mean, std) as lists of floats.
        """
        if data_dir.name == mean_std_name:
            mean_std_file = data_dir
        else:
            mean_std_file = data_dir / mean_std_name
        if mean_std_file.exists():
            with mean_std_file.open('r') as f:
                mean = eval(re.search(r'\[.*\]', f.readline())[0])
                std = eval(re.search(r'\[.*\]', f.readline())[0])
        else:
            dataset = ImageFolder(str(data_dir / 'train'), DataTransformFactory.create_transform('default'))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            mean_tensor, std_tensor = DataStatistics.compute_mean_std(dataloader)
            mean, std = mean_tensor.tolist(), std_tensor.tolist()

            with mean_std_file.open('w') as f:
                f.write(f'mean: {mean}\n')
                f.write(f'std: {std}')

        return mean, std


def save_class_to_idx(data_dir: Path, class_to_idx: Dict[str, int]) -> None:
    """Save the class-to-index mapping to a file.

    Args:
        data_dir: Path to the dataset directory.
        class_to_idx: Dictionary mapping class names to indices.
    """
    with (data_dir / 'class_to_idx.txt').open('w') as f:
        for class_name, idx in class_to_idx.items():
            f.write(f'{class_name} {idx}\n')


def get_embeddings(dataset: Dataset, model: torch.nn.Module) -> Tuple[Tensor, Tensor]:
    """Get embeddings and labels for a dataset using a model.

    Args:
        dataset: The dataset to process.
        model: The model to generate embeddings.

    Returns:
        A tuple of (embeddings, labels) as tensors.
    """
    return testers.BaseTester().get_all_embeddings(dataset, model)


class EarlyStopping:
    """Class to implement early stopping based on validation loss.

    Attributes:
        patience: Number of epochs to wait for improvement (default: 5).
        min_delta: Minimum improvement required to reset counter (default: 0).
        best_loss: Best validation loss observed so far.
        counter: Number of epochs without improvement.
        early_stop: Flag indicating whether to stop training.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """Initialize the EarlyStopping instance.

        Args:
            patience: Number of epochs to wait for improvement (default: 5).
            min_delta: Minimum improvement threshold (default: 0).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss.
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
    """Factory class for creating model instances."""

    _MODEL_TYPES = {
        'EfficientArcFaceModel': EfficientArcFaceModel,
        'DOLG': DOLGModel,
        'DOLGModel': DOLGModel,
        'MLGModel': MLGModel
    }

    @staticmethod
    def create_model(model_structure: str, model_path: str, embedding_size: int) -> torch.nn.Module:
        """Create a model instance based on the specified structure.

        Args:
            model_structure: Name of the model structure.
            model_path: Path to the pretrained model weights.
            embedding_size: Size of the embedding layer.

        Returns:
            The initialized model.

        Raises:
            ValueError: If the model structure is invalid.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_class = ModelFactory._MODEL_TYPES.get(model_structure)
        if not model_class:
            raise ValueError(f"Invalid model structure: {model_structure}")

        model = model_class(pretrained=False, embedding_size=embedding_size).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model


def load_model(model_structure: str, model_path: str, embedding_size: int) -> torch.nn.Module:
    """Load a pre-trained model.

    Args:
        model_structure: The structure of the model to load.
        model_path: Path to the pretrained model file.
        embedding_size: Size of the embedding layer.

    Returns:
        The loaded pre-trained model.
    """
    return ModelFactory.create_model(model_structure, model_path, embedding_size)


def seed_worker(worker_id: int) -> None:
    """Set random seeds for DataLoader workers.

    Args:
        worker_id: ID of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(path: str, dataset_mode: str = 'default', imgsz: int = 224, batch_size: int = 8,
                     shuffle: bool = True, num_workers: int = 8, seed: int = DEFAULT_SEED,
                     mean: List[float] = DEFAULT_MEAN, std: List[float] = DEFAULT_STD,
                     drop_last: bool = False) -> Tuple[DataLoader, Dataset]:
    """Create a DataLoader for a dataset.

    Args:
        path: Path to the dataset directory.
        dataset_mode: Mode of the dataset ('default', 'train', 'val', 'test').
        imgsz: Image size for resizing (default: 224).
        batch_size: Batch size (default: 8).
        shuffle: Whether to shuffle the data (default: True).
        num_workers: Number of worker processes (default: 8).
        seed: Random seed for reproducibility (default: 42).
        mean: Mean values for normalization (default: [0.485, 0.456, 0.406]).
        std: Standard deviation values (default: [0.229, 0.224, 0.225]).
        drop_last: Whether to drop the last incomplete batch (default: False).

    Returns:
        A tuple of (DataLoader, Dataset).
    """
    transform = DataTransformFactory.create_transform(dataset_mode, mean, std, imgsz)
    dataset = ImageFolder(path, transform)
    batch_size = min(batch_size, len(dataset))
    num_devices = torch.cuda.device_count() or 1
    num_workers = min(os.cpu_count() // num_devices, batch_size if batch_size > 1 else 0, num_workers)

    generator = torch.Generator().manual_seed(6148914691236517205 + seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, worker_init_fn=seed_worker, generator=generator, drop_last=drop_last
    ), dataset


class UnNormalize:
    """Class to unnormalize a tensor using mean and standard deviation."""

    def __init__(self, mean: List[float], std: List[float]) -> None:
        """Initialize the UnNormalize transform.

        Args:
            mean: Mean values for each channel.
            std: Standard deviation values for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        """Unnormalize the given tensor.

        Args:
            tensor: The tensor to unnormalize.

        Returns:
            The unnormalized tensor.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def plot_bar_chart(values: List[float], labels: List[str], title: str, ylabel: str, save_path: Path) -> None:
    """Plot and save a bar chart.

    Args:
        values: Values to plot.
        labels: Labels for the bars.
        title: Title of the chart.
        ylabel: Label for the y-axis.
        save_path: Path to save the chart.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(hue=labels, y=values, palette="viridis", legend=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def plot_silhouette_distribution(scores: List[float], save_path: Path) -> None:
    """Plot and save a silhouette score distribution.

    Args:
        scores: Silhouette scores to plot.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(scores, bins=20, kde=True)
    plt.xlabel("Silhouette Score")
    plt.ylabel("Frequency")
    plt.title("Silhouette Score Distribution")
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def get_classes_threshold(dataset, model, save_dir: Path, max_samples_per_class: int = 100):
    embeddings, labels = get_embeddings(dataset, model)
    class_sims, cross_class_sims = compute_class_similarities(embeddings, labels, max_samples_per_class)
    df_thresholds = compute_threshold_stats(class_sims)
    df_cross_stats = compute_cross_class_stats(class_sims, cross_class_sims)  # 傳入 class_sims
    save_dir.mkdir(parents=True, exist_ok=True)
    save_threshold_json(df_thresholds, Path(f"{save_dir}/class_thresholds.json"))
    save_cross_stats_json(df_cross_stats, Path(f"{save_dir}/cross_class_stats.json"))
    plot_similarity_histograms(class_sims, Path(f"{save_dir}/similarity_plots"))
    return df_thresholds, df_cross_stats

def compute_class_similarities(embeddings: torch.Tensor, labels: torch.Tensor, max_samples_per_class: int = 100) -> Tuple[Dict[int, list], Dict[Tuple[int, int], list]]:
    """
    Compute pairwise cosine similarity within and across classes using cosine_similarity.
    
    Args:
        embeddings: Tensor of shape [N, D]
        labels: Tensor of shape [N] or [N, 1]
        max_samples_per_class: Maximum number of samples per class to compute similarities
    
    Returns:
        class_sims: Dict of within-class similarities
        cross_class_sims: Dict of cross-class similarities
    """
    class_sims = defaultdict(list)      # 類別內相似性
    cross_class_sims = defaultdict(list) # 跨類別相似性
    
    if labels.dim() == 2:
        labels = labels.squeeze(1)
    
    # 預先將 unique_classes 轉為整數列表
    unique_classes = [int(cls.item()) for cls in torch.unique(labels)]
    class_embeddings = {}
    
    # 按類別分割並抽樣嵌入
    for cls in unique_classes:
        cls_mask = labels == cls
        cls_embs = embeddings[cls_mask]
        
        # 如果樣本數超過限制，隨機抽樣
        if cls_embs.size(0) > max_samples_per_class:
            indices = torch.randperm(cls_embs.size(0))[:max_samples_per_class]
            cls_embs = cls_embs[indices]
        
        class_embeddings[cls] = cls_embs  # 不在這裡正規化
    
    # 計算類別內相似性
    for cls, emb in class_embeddings.items():
        if emb.size(0) < 2:
            continue
        # 計算所有成對餘弦相似性
        n = emb.size(0)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):  # 避免計算自身相似性
                sim = F.cosine_similarity(emb[i:i+1], emb[j:j+1], dim=1)
                sims.append(sim.item())
        class_sims[cls] = sims
    
    # 計算跨類別相似性
    for i, cls_i in enumerate(unique_classes):
        for cls_j in unique_classes[i + 1:]:  # 只計算上三角
            emb_i, emb_j = class_embeddings[cls_i], class_embeddings[cls_j]
            sims = []
            # 計算 emb_i 和 emb_j 之間的所有成對相似性
            for vec_i in emb_i:
                for vec_j in emb_j:
                    sim = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0), dim=1)
                    sims.append(sim.item())
            cross_class_sims[(cls_i, cls_j)] = sims
    
    return class_sims, cross_class_sims

def compute_threshold_stats(class_sims: Dict[int, list], std_scale: float = 1.0) -> pd.DataFrame:
    """
    Compute statistics for within-class similarities.
    """
    records = []
    for cls, sims in class_sims.items():
        sims = np.array(sims)
        mean_val = sims.mean()
        min_val = sims.min()
        std_val = sims.std()
        threshold_min = mean_val - std_scale * std_val
        threshold_max = mean_val + std_scale * std_val

        records.append({
            "class": cls,
            "mean": round(mean_val, 4),
            "min": round(min_val, 4),
            "std": round(std_val, 4),
            f"threshold_min@1σ": round(threshold_min, 4),
            f"threshold_max@1σ": round(threshold_max, 4),
        })

    df = pd.DataFrame(records)
    return df.sort_values("class").reset_index(drop=True)

def compute_cross_class_stats(class_sims: Dict[int, list], cross_class_sims: Dict[Tuple[int, int], list]) -> Dict[str, pd.DataFrame]:
    """
    Compute statistics for cross-class similarities in a confusion matrix-like format.
    
    Args:
        class_sims: Dict of within-class similarities
        cross_class_sims: Dict of cross-class similarities
    
    Returns:
        Dict containing DataFrames for mean, max, and min similarities
    """
    unique_classes = sorted(set(cls for cls in class_sims.keys()).union(
                            set(cls for pair in cross_class_sims.keys() for cls in pair)))
    n_classes = len(unique_classes)
    
    # 初始化統計矩陣
    mean_matrix = np.zeros((n_classes, n_classes))
    max_matrix = np.zeros((n_classes, n_classes))
    min_matrix = np.zeros((n_classes, n_classes))
    
    # 填充類別內統計
    for cls in unique_classes:
        if cls in class_sims and len(class_sims[cls]) > 0:  # 確保有數據
            sims = np.array(class_sims[cls])
            idx = unique_classes.index(cls)
            mean_matrix[idx, idx] = sims.mean()
            max_matrix[idx, idx] = sims.max()
            min_matrix[idx, idx] = sims.min()
    
    # 填充跨類別統計
    for (cls_i, cls_j), sims in cross_class_sims.items():
        sims = np.array(sims)
        i, j = unique_classes.index(cls_i), unique_classes.index(cls_j)
        mean_matrix[i, j] = mean_matrix[j, i] = sims.mean()
        max_matrix[i, j] = max_matrix[j, i] = sims.max()
        min_matrix[i, j] = min_matrix[j, i] = sims.min()
    
    # 轉為 DataFrame
    df_mean = pd.DataFrame(mean_matrix, index=unique_classes, columns=unique_classes)
    df_max = pd.DataFrame(max_matrix, index=unique_classes, columns=unique_classes)
    df_min = pd.DataFrame(min_matrix, index=unique_classes, columns=unique_classes)
    
    return {"mean": df_mean, "max": df_max, "min": df_min}

def save_threshold_json(df: pd.DataFrame, save_path: Path):
    result_dict = {}
    for _, row in df.iterrows():
        class_id = int(row["class"])
        result_dict[class_id] = {
            "mean": row["mean"],
            "min": row["min"],
            "std": row["std"],
            "threshold_min@1": row["threshold_min@1σ"],
            "threshold_max@1": row["threshold_max@1σ"],
        }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Exported threshold stats to: {save_path}")

def save_cross_stats_json(cross_stats: Dict[str, pd.DataFrame], save_path: Path):
    result_dict = {
        "mean": cross_stats["mean"].to_dict(),
        "max": cross_stats["max"].to_dict(),
        "min": cross_stats["min"].to_dict()
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Exported cross-class stats to: {save_path}")

def plot_similarity_histograms(class_sims: Dict[int, list], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    for cls, sims in class_sims.items():
        plt.figure(figsize=(6, 4))
        plt.hist(sims, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
        plt.title(f"Class {cls} Similarity Distribution")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f"class_{cls}_similarity_hist.png")
        plt.close()
    print(f"Saved all classes similarity plot at: {save_dir}")
