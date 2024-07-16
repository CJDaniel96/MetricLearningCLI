import timm
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn


class ExtractFeaturesModel(nn.Module):
    """
    ExtractFeaturesModel class for extracting features using EfficientNetV2.

    Attributes:
        efficient (torch.nn.Module): EfficientNetV2 model for feature extraction.
        avgpool (torch.nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        flatten (torch.nn.Flatten): Flatten layer.

    Methods:
        forward(x): Forward pass of the model.

    """
    def __init__(self) -> None:
        """
        Initialize the ExtractFeaturesModel instance.
        """
        super().__init__()
        self.efficient = timm.create_model('efficientnetv2_s', pretrained=False, in_chans=3, features_only=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.efficient(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return x


class EfficientArcFaceModel(nn.Module):
    """
    EfficientArcFaceModel class for extracting features using EfficientNetV2 and applying ArcFace.

    Attributes:
        backbone (torch.nn.Module): EfficientNetV2 model for feature extraction.
        neck (torch.nn.Module): Neck layers for further feature processing.
        head (torch.nn.Module): Head layers for ArcFace.

    Methods:
        forward(x): Forward pass of the model.

    """
    def __init__(self, model_name='tf_efficientnetv2_s', pretrained=True, features_only=True, embedding_size=128) -> None:
        """
        Initialize the EfficientArcFaceModel instance.
        """
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
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.backbone(x)
        x = self.neck(x[-1])
        x = self.head(x)

        return x


class TransformerArcFaceModel(nn.Module):
    """
    TransformerArcFaceModel class for extracting features using Vision Transformer and applying ArcFace.

    Attributes:
        backbone (torch.nn.Module): Vision Transformer model for feature extraction.
        head (torch.nn.Module): Head layers for ArcFace.

    Methods:
        forward(x): Forward pass of the model.

    """
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True, num_classes=0, embedding_size=128) -> None:
        """
        Initialize the TransformerArcFaceModel instance.
        """
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.head = nn.Sequential(
            nn.Linear(192, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size), 
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.backbone(x)
        x = self.head(x)

        return x
    

class MultiAtrous(nn.Module):
    def __init__(self, in_channel, out_channel, size, dilation_rates=[3, 6, 9]):
        super().__init__()
        self.dilated_convs = [
            nn.Conv2d(in_channel, int(out_channel/4),
                      kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ]
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, int(out_channel/4), kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(size, size), mode='bilinear')
        )
        self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs)

    def forward(self, x):
        local_feat = []
        for dilated_conv in self.dilated_convs:
            local_feat.append(dilated_conv(x))
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat
    

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)
        projection = torch.bmm(global_feat.unsqueeze(1), torch.flatten(
            local_feat, start_dim=2))
        projection = torch.bmm(global_feat.unsqueeze(
            2), projection).view(local_feat.size())
        projection = projection / \
            (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        return torch.cat([global_feat.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)


class LocalBranch(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=2048, image_size=512):
        super().__init__()
        self.multi_atrous = MultiAtrous(in_channel, hidden_channel, size=int(image_size/8))
        self.conv1x1_1 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        local_feat = self.multi_atrous(x)

        local_feat = self.conv1x1_1(local_feat)
        local_feat = self.relu(local_feat)
        local_feat = self.conv1x1_2(local_feat)
        local_feat = self.bn(local_feat)

        attention_map = self.relu(local_feat)
        attention_map = self.conv1x1_3(attention_map)
        attention_map = self.softplus(attention_map)

        local_feat = F.normalize(local_feat, p=2, dim=1)
        local_feat = local_feat * attention_map

        return local_feat
    

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale_factor=64.0, margin=0.50, criterion=None):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # input is not l2 normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        loss = self.criterion(logit, label)

        return loss, logit
    
    
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    

class DOLGModel(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', pretrained=False, input_dim=3, hidden_dim=1024, embedding_size=128, image_size=224) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=input_dim,
            out_indices=(2, 3)
        )
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = LocalBranch(64, hidden_dim, image_size=image_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM()
        self.fc_1 = nn.Linear(160, hidden_dim)
        self.fc_2 = nn.Linear(int(2*hidden_dim), embedding_size)
        
    def forward(self, x):
        output = self.backbone(x)

        local_feat = self.local_branch(output[-1])  # ,hidden_channel,16,16
        global_feat = self.fc_1(self.gem_pool(output[-2]).squeeze((2, 3)))  # ,1024
        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)

        return feat
    

class MultiheadArcFaceModel(nn.Module):
    """
    EfficientArcFaceModel class for extracting features using EfficientNetV2 and applying ArcFace.

    Attributes:
        backbone (torch.nn.Module): EfficientNetV2 model for feature extraction.
        local_branch (torch.nn.Module): Multi-head self-attention for local features.
        global_branch (torch.nn.Module): Global feature extraction.
        fusion (torch.nn.Module): Orthogonal fusion of local and global features.
        head (torch.nn.Module): Head layers for ArcFace.

    Methods:
        forward(x): Forward pass of the model.
    """
    def __init__(self, model_name='tf_efficientnetv2_s', pretrained=True, features_only=True, embedding_size=128) -> None:
        """
        Initialize the EfficientArcFaceModel instance.
        """
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=features_only)
        
        # Get the output feature shapes by passing a dummy input through the backbone
        dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the input size as needed
        features = self.backbone(dummy_input)
        local_in_channels = features[-1].shape[1]
        global_in_channels = features[-2].shape[1]
        
        # Local branch with multi-head self-attention
        self.local_branch_conv = nn.Sequential(
            nn.Conv2d(local_in_channels, 1280, 1, 1, bias=False), 
            nn.BatchNorm2d(1280, 0.001), 
            nn.SiLU(), 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.local_branch_attention = nn.MultiheadAttention(embed_dim=1280, num_heads=8)
        
        # Global branch
        self.global_branch = nn.Sequential(
            nn.Conv2d(global_in_channels, 1280, 1, 1, bias=False), 
            nn.BatchNorm2d(1280, 0.001), 
            nn.SiLU(), 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Orthogonal fusion
        self.fusion = nn.Linear(2560, 1280)
        
        # Head for final embedding
        self.head = nn.Sequential(
            nn.Linear(1280, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size), 
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        features = self.backbone(x)
        local_features = self.local_branch_conv(features[-1])
        local_features = local_features.unsqueeze(0)  # Add batch dimension for attention
        local_features, _ = self.local_branch_attention(local_features, local_features, local_features)
        local_features = local_features.squeeze(0)  # Remove batch dimension after attention
        
        global_features = self.global_branch(features[-2])
        
        # Orthogonal fusion
        global_feat_norm = torch.norm(global_features, p=2, dim=1, keepdim=True)
        projection = torch.bmm(global_features.unsqueeze(2), torch.flatten(local_features, start_dim=2))
        projection = torch.bmm(global_features.unsqueeze(1), projection).view(local_features.size())
        projection = projection / (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)
        orthogonal_comp = local_features - projection
        global_features = global_features.unsqueeze(-1).unsqueeze(-1)
        fused_features = torch.cat([global_features.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)
        
        # Final embedding
        embedding = self.head(fused_features)

        return embedding