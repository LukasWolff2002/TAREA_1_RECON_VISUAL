"""
Multi-Scale FCOS con Feature Pyramid Network para Query-Conditioned Detection
VERSIÓN CORREGIDA: Con normalización de features para query modulation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Assuming iDoc models are in ./iDoc
# Agregamos la carpeta raíz de iDoc para que encuentre su propio 'utils.py'
sys.path.insert(0, os.path.abspath('./iDoc'))
# Agregamos la carpeta models de iDoc para que encuentre 'vision_transformer'
sys.path.insert(0, os.path.abspath('./iDoc/models'))

import vision_transformer

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network para generar features multi-escala
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # Lateral connections (1x1 convs para reducir dimensionalidad)
        self.lateral_c3 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels, out_channels, 1)
        
        # Top-down pathway (3x3 convs para suavizar upsampling)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # P6 con stride 2 sobre P5
        self.p6_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        self._init_weights()

    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: Dict con C3, C4, C5 (multi-scale features del backbone)
        Returns:
            List de [P3, P4, P5, P6] features
        """
        c3, c4, c5 = features['C3'], features['C4'], features['C5']
        
        # Lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3)
        
        # Top-down pathway con upsampling + add
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        
        # Smooth
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        
        # P6 (stride 64)
        p6 = self.p6_conv(p5)
        
        return [p3, p4, p5, p6]


class Scale(nn.Module):
    """Learnable scale parameter for each FPN level"""
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    
    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    """
    FCOS Detection Head (compartido entre niveles FPN)
    """
    def __init__(self, in_channels, num_classes=1, num_convs=4):
        super().__init__()
        
        # Classification branch
        cls_tower = []
        for i in range(num_convs):
            cls_tower.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        
        # Regression branch
        reg_tower = []
        for i in range(num_convs):
            reg_tower.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            )
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)
        
        # Output layers
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        
        # Scales para cada nivel
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(4)])
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.cls_tower, self.reg_tower]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
        
        # Bias initialization para classification (focal loss)
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_logits.bias, bias_value)
        
        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0)
        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.constant_(self.centerness.bias, 0)
    
    def forward(self, features, level_idx):
        """
        Args:
            features: Feature map (B, C, H, W)
            level_idx: FPN level index (0-3 para P3-P6)
        """
        cls_feat = self.cls_tower(features)
        reg_feat = self.reg_tower(features)
        
        cls_logits = self.cls_logits(cls_feat)
        centerness = self.centerness(reg_feat)
        
        # Bbox regression con scale
        bbox_pred = self.scales[level_idx](self.bbox_pred(reg_feat))
        bbox_pred = F.relu(bbox_pred)  # Distancias positivas
        
        return cls_logits, bbox_pred, centerness


class QueryConditionedFCOS(nn.Module):
    """
    Multi-Scale FCOS condicionado por query sketch
    VERSIÓN CORREGIDA con normalización de features
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone (ViT pre-entrenado)
        self.backbone = vision_transformer.__dict__[config.BACKBONE_TYPE](
            patch_size=config.PATCH_SIZE,
            num_classes=0,
            return_all_tokens=True
        )
        
        # Load pretrained weights
        if os.path.exists(config.BACKBONE_CKPT):
            checkpoint = torch.load(config.BACKBONE_CKPT, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            state_dict = {
                k.replace("module.", "").replace("encoder.model.", ""): v 
                for k, v in state_dict.items()
            }
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded backbone from {config.BACKBONE_CKPT}")
        
        # 🔒 CONGELAR BACKBONE SI ESTÁ CONFIGURADO
        if config.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"🔒 Backbone CONGELADO (no se entrena)")
        else:
            print(f"🔓 Backbone DESCONGELADO (se entrena)")
        
        # Feature extraction adapter para ViT
        self.feature_adapter = nn.ModuleDict({
            'C3': nn.Conv2d(config.FEATURE_DIM, config.FEATURE_DIM, 1),
            'C4': nn.Conv2d(config.FEATURE_DIM, config.FEATURE_DIM, 1),
            'C5': nn.Conv2d(config.FEATURE_DIM, config.FEATURE_DIM, 1),
        })
        
        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels=config.FEATURE_DIM,
            out_channels=config.FPN_OUT_CHANNELS
        )
        
        # FCOS Head
        self.fcos_head = FCOSHead(
            in_channels=config.FPN_OUT_CHANNELS,
            num_classes=1  # Binary: object vs background
        )
        
        # 🆕 CORREGIDO: Query modulation con tanh para limitar rango
        self.query_modulation = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.FPN_OUT_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Linear(config.FPN_OUT_CHANNELS, config.FPN_OUT_CHANNELS),
            nn.Tanh()  # ← Limita el rango a [-1, 1]
        )
    
    def extract_multiscale_features(self, x):
        """
        Extract multi-scale features from ViT backbone
        Simula C3, C4, C5 con diferentes resoluciones
        """
        B, C, H, W = x.shape
        
        # Get ViT patch embeddings
        if self.config.FREEZE_BACKBONE:
            with torch.no_grad():
                features = self.backbone(x)
                if isinstance(features, tuple):
                    features = features[0]
        else:
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[0]
        
        # Descartar token global (CLS)
        if features.dim() == 3:
            features = features[:, 1:, :] 
            
        patch_size = self.config.PATCH_SIZE
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        spatial_features = features.view(B, h_patches, w_patches, -1)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # Create pyramid usando pooling
        c3 = self.feature_adapter['C3'](spatial_features)
        c4 = self.feature_adapter['C4'](F.avg_pool2d(spatial_features, 2))
        c5 = self.feature_adapter['C5'](F.avg_pool2d(spatial_features, 4))
        
        return {'C3': c3, 'C4': c4, 'C5': c5}
    
    def forward(self, images, sketches):
        """
        Args:
            images: (B, 3, H, W)
            sketches: (B, 3, 224, 224)
        Returns:
            predictions: List de dicts con cls, reg, ctr para cada nivel
        """
        B = images.shape[0]
        
        # Extract query features
        if self.config.FREEZE_BACKBONE:
            with torch.no_grad():
                sketch_features = self.backbone(sketches)
                if isinstance(sketch_features, tuple):
                    sketch_features = sketch_features[0]
        else:
            sketch_features = self.backbone(sketches)
            if isinstance(sketch_features, tuple):
                sketch_features = sketch_features[0]
        
        # Global pooling
        if sketch_features.dim() > 2:
            sketch_features = sketch_features.mean(dim=1)  # (B, D)
        
        # 🆕 CRÍTICO: Normalizar features del sketch (como en tu código que funcionaba)
        sketch_features = F.normalize(sketch_features, p=2, dim=-1)
        
        # Query modulation weights (ya limitado por tanh)
        modulation = self.query_modulation(sketch_features)  # (B, FPN_OUT_CHANNELS)
        
        # Extract multi-scale image features
        image_features = self.extract_multiscale_features(images)
        
        # FPN
        pyramid_features = self.fpn(image_features)  # [P3, P4, P5, P6]
        
        # Apply query modulation y FCOS head
        predictions = []
        for level_idx, feat in enumerate(pyramid_features):
            # 🆕 CRÍTICO: Normalizar features de imagen antes de modular
            feat_norm = F.normalize(feat, p=2, dim=1)
            
            # Modulation weights (B, C) -> (B, C, 1, 1)
            modulation_weights = modulation.unsqueeze(-1).unsqueeze(-1)
            
            # 🆕 CORREGIDO: Modulación más suave (aditiva + escala pequeña)
            # En lugar de multiplicar directamente, sumamos con factor pequeño
            feat_modulated = feat_norm + modulation_weights * 0.2
            
            # Re-escalar para que tenga magnitud similar a feat original
            feat_modulated = feat_modulated * feat.norm(dim=1, keepdim=True).mean()
            
            # FCOS predictions
            cls_logits, bbox_pred, centerness = self.fcos_head(feat_modulated, level_idx)
            
            predictions.append({
                'cls': cls_logits,      # (B, 1, H, W)
                'reg': bbox_pred,       # (B, 4, H, W)
                'ctr': centerness,      # (B, 1, H, W)
                'stride': self.config.FPN_STRIDES[level_idx]
            })
        
        return predictions


def build_model(config):
    """Build and return the model"""
    model = QueryConditionedFCOS(config)
    
    # Contar parámetros entrenables
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {frozen_params:,}")
    
    return model