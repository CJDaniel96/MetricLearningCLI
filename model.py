import timm
from torch import nn


class ExtractFeaturesModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficient = timm.create_model('efficientnetv2_s', pretrained=False, in_chans=3, features_only=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.efficient(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return x


class EfficientArcFaceModel(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', pretrained=True, features_only=True, embedding_size=128) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=features_only)
        self.neck = nn.Sequential(
            nn.Conv2d(256, 1280, 1, 1, bias=False), 
            nn.BatchNorm2d(1280, 0.001), 
            nn.SiLU(), 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(1280, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size), 
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[-1])
        x = self.head(x)

        return x


class TransformerArcFaceModel(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True, num_classes=0, embedding_size=128) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.head = nn.Sequential(
            nn.Linear(192, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size), 
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x