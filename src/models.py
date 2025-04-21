import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel

class CompositeModel(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.backbone_type = backbone
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:  # ViT
            self.backbone = ViTModel.from_pretrained(
                'google/vit-base-patch16-224'
            )
            num_ftrs = self.backbone.config.hidden_size
            
    
            # freeze_layers = True
            # if freeze_layers:
            #     for name, param in self.backbone.named_parameters():
            #         param.requires_grad = False
            #         # Example: Unfreeze last two transformer blocks
            #         if any(f'encoder.layer.{i}.' in name for i in [10, 11]):
            #             param.requires_grad = True
            #         # Ensure pooler layer is trainable if used
            #         elif 'pooler' in name:
            #             param.requires_grad = True
        
        self.head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2 + 200)  # room_count, wall_count, 100 wall coordinates
        )

    def forward(self, x):
        if self.backbone_type == 'resnet50':
            features = self.backbone(x)
        else:  # ViT
            outputs = self.backbone(x)
            features = outputs.last_hidden_state[:, 0]

        if features is None:
            raise ValueError(f"Features extracted from {self.backbone_type} backbone are None.")

        output = self.head(features)
        return {
            'room_count': output[:, 0],
            'wall_count': output[:, 1],
            'wall_coords': output[:, 2:]
        } 