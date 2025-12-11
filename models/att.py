import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class ChannelAttention(nn.Module):

    #Channel attention 
    def __init__(self, channels, reduction=16):
        super().__init__()
        #  bottleneck the channel through mlp vector C -> hidden -> C
        # reduction controls how small the hidden layer is.
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        # how strong is each channel overall
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c) #avg
        mx_map = F.adaptive_max_pool2d(x, 1).view(b, c) #max
        
        # Take max over channels for each sample, then broadcast back
        mx, _ = mx_map.max(dim=1, keepdim=False) 
        mx = mx.view(b, 1).expand(b, c) 

        #feed through mlp gate
        gate = self.mlp(avg + mx)
        gate = torch.sigmoid(gate).view(b, c, 1, 1)
        return x * gate


class SpatialAttention(nn.Module):
       #Now we want a heatmap sort of
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        # 2 input channels: avg and max over original channels
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        att = torch.sigmoid(self.conv(cat)) 
        return x * att #reweight pixels


class AttBackboneWithFPN(nn.Module):
    #Attention-augmented ResNet+FPN backbone.
    #We take the standard ResNet+FPN backbone from torchvision.
    #For each FPN level (P2..P5), we:
    #      1) Decide which channels are important for this image (ChannelAttention).
    #      2) Decide which spatial locations are important (SpatialAttention).
    # Mask R-CNN then builds proposals and heads on these 'cleaner' feature maps.

    def __init__(self, backbone_fpn):
        super().__init__()
        self.body = backbone_fpn.body 
        self.fpn = backbone_fpn.fpn
        self.out_channels = backbone_fpn.out_channels

        # One shared attention module reused for all pyramid levels
        self.ca = ChannelAttention(self.out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        feats = self.body(x) 
        feats = self.fpn(feats)
        out = {}
        for name, fmap in feats.items():
            y = self.ca(fmap)
            y = self.sa(y)
            out[name] = y
        return out #dict of feature maps


def maskrcnn_attfpn(num_classes,weights_backbone=True,trainable_backbone_layers=3):
      #We only insert attention right after the FPN, before the RPN/ROI heads.

    # pyramid level (P6):
    # backbone_fpn = resnet_fpn_backbone(
    #     "resnet50",
    #     weights_backbone=weights_backbone,
    #     trainable_backbone_layers=trainable_backbone_layers,
    #     extra_blocks=LastLevelMaxPool(),
    # )
    #backbone_fpn = resnet_fpn_backbone("resnet50",weights_backbone=weights_backbone,trainable_backbone_layers=trainable_backbone_layers) #this should be a param perhaps

    backbone_fpn = resnet_fpn_backbone("resnet50",pretrained=weights_backbone,trainable_layers=trainable_backbone_layers) #this is something with pytorch versions
    
    backbone = AttBackboneWithFPN(backbone_fpn)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model
