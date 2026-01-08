import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DeiTModel

class HybridCNNTransformerEncoder(nn.Module):
    """Hybrid: ResNet50 (CNN) + DeiT-Tiny (Transformer)"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # CNN: ResNet50
        self.cnn_encoder = models.resnet50(pretrained=pretrained)
        self.cnn_encoder = nn.Sequential(*list(self.cnn_encoder.children())[:-1])
        self.cnn_feature_dim = 2048
        
        # Transformer: DeiT-Tiny
        if pretrained:
            self.transformer_encoder = DeiTModel.from_pretrained(
                'facebook/deit-tiny-patch16-224'
            )
        self.transformer_feature_dim = 192
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + self.transformer_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512)
        )
        
        self.output_dim = 512
    
    def forward(self, x):
        cnn_features = self.cnn_encoder(x)
        cnn_features = cnn_features.squeeze(-1).squeeze(-1)
        
        transformer_output = self.transformer_encoder(x)
        transformer_features = transformer_output.last_hidden_state[:, 0]
        
        combined = torch.cat([cnn_features, transformer_features], dim=1)
        fused = self.fusion(combined)
        
        return fused


class MultiPlaneHybridFusion(nn.Module):
    """Multi-plane with Hybrid encoders"""
    
    def __init__(self, num_classes=1, dropout_rate=0.4):
        super().__init__()
        
        print("��️  Building Hybrid CNN-Transformer Multi-Plane Model...")
        
        print("  📦 Sagittal: ResNet50 + DeiT-Tiny")
        self.sagittal_encoder = HybridCNNTransformerEncoder(pretrained=True)
        
        print("  📦 Coronal: ResNet50 + DeiT-Tiny")
        self.coronal_encoder = HybridCNNTransformerEncoder(pretrained=True)
        
        print("  📦 Axial: ResNet50 + DeiT-Tiny")
        self.axial_encoder = HybridCNNTransformerEncoder(pretrained=True)
        
        feature_dim = 512
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        print("✅ Model built!")
        self._print_info()
    
    def forward(self, sagittal, coronal, axial):
        sag_features = self.sagittal_encoder(sagittal)
        cor_features = self.coronal_encoder(coronal)
        axi_features = self.axial_encoder(axial)
        
        multi_plane_features = torch.stack([sag_features, cor_features, axi_features], dim=1)
        
        attended_features, _ = self.cross_attention(
            multi_plane_features,
            multi_plane_features,
            multi_plane_features
        )
        
        pooled_features = attended_features.mean(dim=1)
        fused = self.fusion_layer(pooled_features)
        logits = self.classifier(fused)
        
        return logits
    
    def get_features(self, sagittal, coronal, axial):
        sag_features = self.sagittal_encoder(sagittal)
        cor_features = self.coronal_encoder(coronal)
        axi_features = self.axial_encoder(axial)
        
        multi_plane_features = torch.stack([sag_features, cor_features, axi_features], dim=1)
        attended_features, _ = self.cross_attention(
            multi_plane_features,
            multi_plane_features,
            multi_plane_features
        )
        
        pooled_features = attended_features.mean(dim=1)
        fused = self.fusion_layer(pooled_features)
        
        return fused
    
    def _print_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*70}")
        print("MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"• 3× ResNet50 + 3× DeiT-Tiny")
        print(f"• Cross-plane attention + Fusion")
        print(f"• Total params: {total:,}")
        print(f"• Trainable: {trainable:,}")
        print(f"{'='*70}")


if __name__ == '__main__':
    print("\n🧪 Testing Model...")
    model = MultiPlaneHybridFusion(num_classes=1)
    
    sag = torch.randn(2, 3, 224, 224)
    cor = torch.randn(2, 3, 224, 224)
    axi = torch.randn(2, 3, 224, 224)
    
    output = model(sag, cor, axi)
    print(f"\n✅ Output: {output.shape}")
    print("✅ Model works!")
