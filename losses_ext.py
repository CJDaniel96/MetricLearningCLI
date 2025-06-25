import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import SubCenterArcFaceLoss, TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner


def is_hybrid_loss(criterion):
    return (
        hasattr(criterion, "triplet")
        and hasattr(criterion, "cosface")
        and hasattr(criterion, "center")
    )


class HybridMarginLoss(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        subcenter_margin: float = 0.4,
        subcenter_scale: float = 30.0,
        sub_centers: int = 3,
        triplet_margin: float = 0.3,
        center_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.subcenter_arcface = SubCenterArcFaceLoss(
            embedding_size=embedding_size,
            num_classes=num_classes,
            sub_centers=sub_centers,
            margin=subcenter_margin,
            scale=subcenter_scale,
        )
        self.triplet_loss = TripletMarginLoss(margin=triplet_margin)
        self.center_loss_weight = center_loss_weight
        self.center_loss_centers = nn.Parameter(
            torch.randn(num_classes, embedding_size)
        )

    def forward(self, embeddings, labels):
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Sub-center ArcFace loss
        loss_arc = self.subcenter_arcface(embeddings, labels)

        # Triplet loss (batch hard mining)
        miner = BatchHardMiner()
        hard_triplets = miner(embeddings, labels)
        loss_triplet = self.triplet_loss(embeddings, labels, hard_triplets)

        # Center loss
        centers_batch = self.center_loss_centers[labels]
        loss_center = ((embeddings - centers_batch) ** 2).sum(dim=1).mean()

        # Combine all
        total_loss = (
            loss_arc + 0.5 * loss_triplet + self.center_loss_weight * loss_center
        )
        return total_loss
