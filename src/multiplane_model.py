import torch
import torch.nn as nn
import torchvision.models as models

class MultiPlaneFusion(nn.Module):
    """
    INNOVATION: Fuse information from all 3 MRI planes
    Architecture:
    1. Separate ResNet50 encoders for each plane
    2. Cross-plane attention mechanism
    3. Feature fusion and classification
    """
    
    def __init__(self, num_classes=1, dropout_rate=0.4):
        super(MultiPlaneFusion, self).__init__()
        
        print("🔨 Creating Multi-Plane Fusion Model...")
        
        # Separate encoders for each plane
        self.sagittal_encoder = self._create_encoder()
        self.coronal_encoder = self._create_encoder()
        self.axial_encoder = self._create_encoder()
        
        print("✅ Created 3 plane-specific encoders")
        
        # Feature dimension from ResNet50
        feature_dim = 2048
        
        # Projection layers
        self.sagittal_proj = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.coronal_proj = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.axial_proj = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-plane attention
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=False
        )
        
        print("✅ Created cross-plane attention mechanism")
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(256, num_classes)
        )
        
        print("✅ Multi-Plane Fusion Model created successfully!")
        self._count_parameters()
    
    def _create_encoder(self):
        """Create ResNet50 encoder"""
        resnet = models.resnet50(pretrained=True)
        # Remove final FC layer
        modules = list(resnet.children())[:-1]
        encoder = nn.Sequential(*modules)
        
        # Freeze early layers
        for param in list(encoder.parameters())[:-20]:
            param.requires_grad = False
        
        return encoder
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total:,}")
        print(f"📊 Trainable parameters: {trainable:,}")
    
    def forward(self, sagittal, coronal, axial):
        # Extract features from each plane
        sag_feat = self.sagittal_encoder(sagittal).flatten(1)
        cor_feat = self.coronal_encoder(coronal).flatten(1)
        axi_feat = self.axial_encoder(axial).flatten(1)
        
        # Project to common dimension
        sag_proj = self.sagittal_proj(sag_feat)
        cor_proj = self.coronal_proj(cor_feat)
        axi_proj = self.axial_proj(axi_feat)
        
        # Stack for attention: (3, batch, 512)
        features = torch.stack([sag_proj, cor_proj, axi_proj], dim=0)
        
        # Cross-plane attention
        attended, _ = self.attention(features, features, features)
        
        # Concatenate all planes
        fused = torch.cat([attended[0], attended[1], attended[2]], dim=1)
        
        # Classify
        output = self.classifier(fused)
        
        return output
