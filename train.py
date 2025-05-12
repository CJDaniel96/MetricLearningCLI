import json
import torch
import pandas as pd
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import silhouette_score
from model import DOLGModel, EfficientArcFaceModel, MLGModel, MLGModelV2
from losses_ext import HybridMarginLoss, is_hybrid_loss
from utils import EarlyStopping, setup_seed, create_dataloader, DataStatistics, save_class_to_idx, get_embeddings, plot_bar_chart, plot_silhouette_distribution, get_classes_threshold
from evaluate import evaluate_model_on_testset


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    data_dir: str
    save_dir: str
    epochs: int = 40
    batch_size: int = 64
    image_size: int = 224
    num_classes: int = None
    num_workers: int = 4
    embedding_size: int = 128
    learning_rate: float = 1e-3
    loss_learning_rate: float = 1e-4
    seed: int = 0
    early_stop_patience: Optional[int] = 3
    pretrained_weights: Optional[str] = None
    model_type: str = "EfficientArcFaceModel"
    loss_type: str = "SubCenterArcFaceLoss"
    optimizer_type: str = "Adam"

    def save_config(self) -> None:
        """Save the training configuration to a JSON file.

        The configuration is written to 'opts.json' in the save directory.
        """
        save_path = Path(self.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with save_path.joinpath('opts.json').open('w') as file:
            json.dump(self.__dict__, file, indent=2)

class ModelFactory:
    """Factory class for creating model and loss instances."""
    
    @staticmethod
    def create_model(config: TrainingConfig, device: str) -> torch.nn.Module:
        """Create a model instance based on the configuration.

        Args:
            config: TrainingConfig object containing model parameters.
            device: The device to place the model on (e.g., 'cuda' or 'cpu').

        Returns:
            A PyTorch model instance.

        Raises:
            ValueError: If the model type is not supported.
        """
        model_classes = {
            "EfficientArcFaceModel": EfficientArcFaceModel,
            "DOLGModel": DOLGModel,
            "MLGModel": MLGModel,
            "MLGModelV2": MLGModelV2
        }
        
        model_class = model_classes.get(config.model_type)
        if not model_class:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        model_params = {"embedding_size": config.embedding_size}
        if config.model_type == "DOLGModel":
            model_params["image_size"] = config.image_size

        model = model_class(**model_params).to(device)
        
        if config.pretrained_weights:
            model.load_state_dict(torch.load(config.pretrained_weights))
        
        return model

    @staticmethod
    def create_loss(config: TrainingConfig, device: str) -> torch.nn.Module:
        """Create a loss function instance based on the configuration.

        Args:
            config: TrainingConfig object containing loss parameters.
            device: The device to place the loss function on (e.g., 'cuda' or 'cpu').

        Returns:
            A PyTorch loss function instance.

        Raises:
            ValueError: If the loss type is not supported.
        """
        loss_classes = {
            "SubCenterArcFaceLoss": losses.SubCenterArcFaceLoss,
            "ArcFaceLoss": losses.ArcFaceLoss,
            "HybridMarginLoss": HybridMarginLoss
        }
        
        loss_class = loss_classes.get(config.loss_type)
        if not loss_class:
            raise ValueError(f"Unsupported loss type: {config.loss_type}")
            
        return loss_class(
            num_classes=config.num_classes,
            embedding_size=config.embedding_size
        ).to(device)

class Trainer:
    """Class responsible for managing the training process."""
    
    def __init__(self, config: TrainingConfig, device: str):
        """Initialize the Trainer with configuration and device.

        Args:
            config: TrainingConfig object with training parameters.
            device: The device to use for training (e.g., 'cuda' or 'cpu').
        """
        self.config = config
        self.device = device
        self.save_dir = Path(config.save_dir)
        self.writer = SummaryWriter(self.save_dir / 'logs')
        if os.name == 'nt' and len(config.data_dir) > 260:
            print("Warning: Data Directory Path length exceeds 260 characters. Consider enabling long path support on Windows.")
        self.data_dir = Path(config.data_dir)
        if os.name == 'nt' and len(config.save_dir) > 260:
            print("Warning: Save Directory Path length exceeds 260 characters. Consider enabling long path support on Windows.")
        self._setup_training()

    def _setup_training(self) -> None:
        """Set up the training environment, including data loaders and model components."""
        setup_seed(self.config.seed)
        
        mean, std = DataStatistics.get_mean_std(self.data_dir, self.config.batch_size)
        self.train_loader, self.train_dataset = create_dataloader(
            str(self.data_dir / 'train'), 'train', self.config.image_size,
            self.config.batch_size, True, self.config.num_workers,
            self.config.seed, mean, std, drop_last=True
        )
        self.val_loader, self.val_dataset = create_dataloader(
            str(self.data_dir / 'val'), 'train', self.config.image_size,
            self.config.batch_size, False, self.config.num_workers,
            self.config.seed, mean, std
        )
        self.config.num_classes = len(self.train_dataset.classes)
        save_class_to_idx(self.data_dir, self.train_dataset.class_to_idx)

        self.model = ModelFactory.create_model(self.config, self.device)
        self.criterion = ModelFactory.create_loss(self.config, self.device)
        self.optimizer, self.loss_optimizer = self._create_optimizers()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        self.loss_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.loss_optimizer, T_max=1000)
        self.accuracy_calculator = AccuracyCalculator(include=('AMI', 'mean_average_precision', 'mean_reciprocal_rank', 'precision_at_1'), k=10)
        self.early_stopping = EarlyStopping(patience=self.config.early_stop_patience) \
            if self.config.early_stop_patience else None

    def _create_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Create optimizers for the model and loss function.

        Returns:
            A tuple of two optimizers: one for the model and one for the loss function.

        Raises:
            ValueError: If the optimizer type is not supported.
        """
        optimizers = {
            "Adam": lambda params, lr: torch.optim.Adam(params, lr=lr),
            "SGD": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-5)
        }
        
        optimizer_class = optimizers.get(self.config.optimizer_type)
        if not optimizer_class:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
            
        return (
            optimizer_class(self.model.parameters(), self.config.learning_rate),
            optimizer_class(self.criterion.parameters(), self.config.loss_learning_rate)
        )

    def train(self) -> None:
        best_loss = float('inf')

        for epoch in range(self.config.epochs):
            print(f'📘 Epoch {epoch + 1}/{self.config.epochs}')
            print('-' * 40)

            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self._log_metrics(epoch, train_loss, val_loss, best_model=False)

            if val_loss < best_loss:
                best_loss = val_loss
                self._save_model('best.pt', val_loss)
                self._log_metrics(epoch, train_loss, val_loss, best_model=True)

            self._save_checkpoint(epoch, train_loss)

            print(f"🔍 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_loss:.4f}\n")

            if self._should_early_stop(val_loss):
                print("🛑 Early stopping triggered.")
                break

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for inputs, labels in tqdm(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            self.loss_optimizer.zero_grad()

            embeddings = self.model(inputs)

            if is_hybrid_loss(self.criterion):
                pairs = self.criterion.miner(embeddings, labels)
                loss = self.criterion.triplet(embeddings, labels, pairs) + \
                       self.criterion.cosface(embeddings, labels) + \
                       self.criterion.center_weight * self.criterion.center(embeddings, labels)
            else:
                loss = self.criterion(embeddings, labels)

            loss.backward()
            self.optimizer.step()
            self.loss_optimizer.step()
            self.scheduler.step()
            self.loss_scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                embeddings = self.model(inputs)

                if is_hybrid_loss(self.criterion):
                    loss = self.criterion.cosface(embeddings, labels) + \
                           self.criterion.triplet(embeddings, labels, None) + \
                           self.criterion.center_weight * self.criterion.center(embeddings, labels)
                else:
                    loss = self.criterion(embeddings, labels)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float, best_model: bool = False) -> None:
        """Log training metrics and save them to CSV and visualizations.

        Args:
            epoch: The current epoch number (0-based).
            train_loss: The average training loss for the epoch.
            val_loss: The average validation loss for the epoch.
        """
        train_embeddings, train_labels = get_embeddings(self.train_dataset, self.model)
        val_embeddings, val_labels = get_embeddings(self.val_dataset, self.model)
        
        accuracies = self.accuracy_calculator.get_accuracy(
            query=val_embeddings,
            query_labels=val_labels.squeeze(1),
            reference=train_embeddings,
            reference_labels=train_labels.squeeze(1)
        )
        
        silhouette_scores = silhouette_score(val_embeddings.cpu().numpy(), val_labels.cpu().numpy())
        
        metrics_data = {
            'Epoch': epoch + 1,
            'Train_Loss': train_loss,
            'Val_Loss': val_loss,
            'Silhouette_Score': silhouette_scores
        }
        
        metrics_data.update(accuracies)
        
        if not best_model:
            self._save_metrics_to_csv(metrics_data)
                
            print(f'[{epoch + 1:03d}/{self.config.epochs:03d}] '
                f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
            
            self.writer.add_scalar('Loss/train', train_loss, epoch + 1)
            self.writer.add_scalar('Loss/val', val_loss, epoch + 1)
            for metric, value in accuracies.items():
                self.writer.add_scalar(f'Accuracy/{metric}', value, epoch + 1)
                print(f'{metric}: {value:.4f}')
            print(f"Silhouette Score: {silhouette_scores:.4f}")
        
        if best_model:
            plot_bar_chart(
                list(accuracies.values()), 
                list(accuracies.keys()), 
                "Metric Learning Evaluation", 
                "Score", 
                self.save_dir / "metrics_bar_chart.png"
            )
            
            plot_silhouette_distribution(
                [silhouette_scores], 
                self.save_dir / "silhouette_distribution.png"
            )
            
            df_thresholds, df_cross_stats = get_classes_threshold(self.val_dataset, self.model, self.save_dir / "thresholds")

    def _save_thresholds_to_json(self, thresholds_data: Dict[str, float], thresholds_file: str, folder_name: str = "thresholds") -> None:
        """Save class-specific thresholds to a JSON file.

        Args:
            thresholds_data: Dictionary mapping class IDs to their thresholds.
            thresholds_file: Name of the file to save thresholds to.
            folder_name: Subdirectory name within data_dir to store thresholds (default: 'thresholds').
        """
        json_path = self.save_dir / folder_name / thresholds_file
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(thresholds_data, f, indent=2)
        
    def _save_metrics_to_csv(self, metrics_data: Dict[str, float]) -> None:
        """Save training metrics to a CSV file.

        Args:
            metrics_data: Dictionary containing metrics for the current epoch.
        """
        csv_path = self.save_dir / "training_metrics.csv"
        
        df = pd.DataFrame([metrics_data])
        
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)

    def _save_model(self, filename: str, loss: float) -> None:
        """Save the model state to a file.

        Args:
            filename: Name of the file to save the model to.
            loss: The validation loss associated with this model save.
        """
        torch.save(self.model.state_dict(), self.save_dir / filename)
        print(f'Saving model with loss {loss/len(self.val_loader):.6f}\n')

    def _save_checkpoint(self, epoch: int, train_loss: float) -> None:
        """Save a checkpoint of the model state.

        Args:
            epoch: The current epoch number (0-based).
            train_loss: The average training loss for the epoch.
        """
        checkpoint_path = self.save_dir / f'Epoch_{epoch+1}_Loss_{train_loss:.6f}.pt'
        torch.save(self.model.state_dict(), str(checkpoint_path))

    def _should_early_stop(self, val_loss: float) -> bool:
        """Check if training should stop early based on validation loss.

        Args:
            val_loss: The validation loss for the current epoch.

        Returns:
            True if early stopping is triggered, False otherwise.
        """
        if self.early_stopping:
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                return True
        return False

def parse_arguments() -> TrainingConfig:
    """Parse command-line arguments and return a TrainingConfig object.

    Returns:
        A TrainingConfig instance populated with parsed arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Model Training Configuration')
    
    parser.add_argument('--data-dir', required=True, help='Dataset directory')
    parser.add_argument('--save-dir', required=True, help='Save directory')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--loss-learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--early-stop-patience', type=int, default=3)
    parser.add_argument('--pretrained-weights', type=str, default=None)
    parser.add_argument('--model-type', default='EfficientArcFaceModel')
    parser.add_argument('--loss-type', default='SubCenterArcFaceLoss')
    parser.add_argument('--optimizer-type', default='Adam')
    
    args = parser.parse_args()
    return TrainingConfig(**vars(args))

def main():
    """Main function to execute the training process."""
    config = parse_arguments()
    config.save_config()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    trainer = Trainer(config, device)
    trainer.train()

if __name__ == '__main__':
    main()