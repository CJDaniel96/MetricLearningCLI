# evaluate.py
import torch
from pytorch_metric_learning.utils.inference import InferenceModel
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import json

def evaluate_model_on_testset(model, test_dataset, save_dir: str, batch_size=64, device='cuda'):
    model.eval()
    model = model.to(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    inference_model = InferenceModel(model=model)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            embeddings = inference_model(imgs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    accuracy_calculator = AccuracyCalculator(
        include=('precision_at_1', 'mean_average_precision', 'mean_reciprocal_rank'), k=10
    )
    metrics = accuracy_calculator.get_accuracy(
        embeddings, labels, embeddings, labels
    )
    metrics['silhouette_score'] = silhouette_score(embeddings.numpy(), labels.numpy())

    # Save results
    with open(save_path / "test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(save_path / "test_metrics.csv", index=False)

    print(f"✅ Evaluation complete. Saved to {save_path}")
    return metrics
