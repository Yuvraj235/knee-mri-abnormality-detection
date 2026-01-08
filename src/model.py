import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DeiTModel
import warnings
warnings.filterwarnings('ignore')

class ImprovedResNetDeiTFusion(nn.Module):
    """Improved fusion model with better regularization"""
    
    def __init__(self, num_classes=1, fusion_type='concat', pretrained=True, dropout_rate=0.5):
        super(ImprovedResNetDeiTFusion, self).__init__()
        
        self.fusion_type = fusion_type
        
        # ResNet50 backbone
        print("📥 Loading ResNet50 (pretrained)...")
        self.resnet = models.resnet50(pretrained=pretrained)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Freeze early ResNet layers
        for param in list(self.resnet.parameters())[:-30]:
            param.requires_grad = False
        
        print("✅ ResNet50 loaded (early layers frozen)")
        
        # DeiT-Tiny backbone
        print("📥 Loading DeiT-Tiny (this may take a minute)...")
        self.deit = DeiTModel.from_pretrained('facebook/deit-tiny-patch16-224')
        deit_features = self.deit.config.hidden_size
        
        # Freeze early DeiT layers
        for param in list(self.deit.parameters())[:-40]:
            param.requires_grad = False
        
        print("✅ DeiT-Tiny loaded (early layers frozen)")
        
        # Projection layers with BatchNorm
        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.deit_proj = nn.Sequential(
            nn.Linear(deit_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion and classification
        if fusion_type == 'concat':
            fusion_dim = 1024
        elif fusion_type == 'add':
            fusion_dim = 512
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(128, num_classes)
        )
        
        print("✅ Model architecture created successfully")
    
    def forward(self, x):
        # ResNet features
        resnet_feat = self.resnet(x)
        resnet_feat = self.resnet_proj(resnet_feat)
        
        # DeiT features
        deit_output = self.deit(x)
        deit_feat = deit_output.last_hidden_state[:, 0]  # CLS token
        deit_feat = self.deit_proj(deit_feat)
        
        # Fusion
        if self.fusion_type == 'concat':
            fused_feat = torch.cat([resnet_feat, deit_feat], dim=1)
        elif self.fusion_type == 'add':
            fused_feat = resnet_feat + deit_feat
        
        # Classification
        output = self.classifier(fused_feat)
        return output

# For compatibility
ResNetDeiTFusion = ImprovedResNetDeiTFusion
