"""
Script de Debugging: Diagnóstico del Loss y Targets
Verifica por qué el modelo no está aprendiendo
"""
import os
import sys
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from data.dataset import create_dataloaders
from models.fcos_model import build_model
from models.loss import FCOSLoss


def analyze_batch(images, sketches, targets, model, criterion, device):
    """Analizar un batch en detalle"""
    
    images = images.to(device)
    sketches = sketches.to(device)
    
    print("\n" + "="*80)
    print("BATCH ANALYSIS")
    print("="*80)
    
    # 1. Verificar inputs
    print(f"\n📥 INPUTS:")
    print(f"  Batch size: {images.shape[0]}")
    print(f"  Image shape: {images.shape}")
    print(f"  Sketch shape: {sketches.shape}")
    
    # 2. Verificar targets
    print(f"\n🎯 TARGETS:")
    for idx, target in enumerate(targets):
        boxes = target['boxes']
        print(f"  Image {idx}: {len(boxes)} boxes")
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sizes = torch.sqrt(areas)
            print(f"    Box sizes (min/mean/max): {sizes.min():.1f} / {sizes.mean():.1f} / {sizes.max():.1f}")
            
            # Verificar asignación FPN
            for box_idx, (box, size) in enumerate(zip(boxes, sizes)):
                assigned_level = None
                for level_idx, (min_s, max_s) in enumerate(Config.FPN_SIZE_RANGES):
                    if min_s <= size < max_s:
                        assigned_level = f"P{level_idx+3}"
                        break
                print(f"      Box {box_idx}: size={size:.1f}px → {assigned_level if assigned_level else 'NINGUNO!'}")
    
    # 3. Forward pass
    print(f"\n🔮 FORWARD PASS:")
    with torch.no_grad():
        predictions = model(images, sketches)
    
    print(f"  Number of FPN levels: {len(predictions)}")
    for level_idx, pred in enumerate(predictions):
        print(f"  Level {level_idx} (P{level_idx+3}):")
        print(f"    cls shape: {pred['cls'].shape}")
        print(f"    reg shape: {pred['reg'].shape}")
        print(f"    ctr shape: {pred['ctr'].shape}")
        print(f"    stride: {pred['stride']}")
        
        # Estadísticas de predicciones
        cls_probs = torch.sigmoid(pred['cls'])
        print(f"    cls probs (min/mean/max): {cls_probs.min():.4f} / {cls_probs.mean():.4f} / {cls_probs.max():.4f}")
    
    # 4. Loss calculation
    print(f"\n💥 LOSS CALCULATION:")
    loss_dict = criterion(predictions, targets)
    
    # Extraer valores de forma segura
    cls_val = loss_dict['cls'].item() if torch.is_tensor(loss_dict['cls']) else loss_dict['cls']
    reg_val = loss_dict['reg'].item() if torch.is_tensor(loss_dict['reg']) else loss_dict['reg']
    ctr_val = loss_dict['ctr'].item() if torch.is_tensor(loss_dict['ctr']) else loss_dict['ctr']
    
    print(f"  Total loss: {loss_dict['total'].item():.6f}")
    print(f"  Cls loss:   {cls_val:.6f}")
    print(f"  Reg loss:   {reg_val:.6f}")
    print(f"  Ctr loss:   {ctr_val:.6f}")
    print(f"  Num pos:    {loss_dict.get('num_pos', 'N/A')}")
    
    # 5. Verificar targets preparados
    print(f"\n🎲 TARGET PREPARATION:")
    level_targets = criterion.prepare_targets(predictions, targets)
    
    for level_idx, batch_targets in enumerate(level_targets):
        stride = predictions[level_idx]['stride']
        h, w = predictions[level_idx]['cls'].shape[-2:]
        
        # Grid bounds
        grid_max_x = w * stride
        grid_max_y = h * stride
        
        print(f"\n  Level {level_idx} (P{level_idx+3}, stride {stride}):")
        print(f"    Grid size: {h}x{w} → covers [0, {grid_max_x}] × [0, {grid_max_y}] px")
        
        total_pos = 0
        for batch_idx, target in enumerate(batch_targets):
            mask = target['mask']
            num_pos = mask.sum().item()
            total_pos += num_pos
            
            # Obtener boxes originales de este batch
            orig_boxes = targets[batch_idx]['boxes']
            
            if len(orig_boxes) > 0:
                print(f"    Image {batch_idx}: {num_pos} positive pixels")
                
                # Verificar cada box
                for box_idx, box in enumerate(orig_boxes):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # Check si está dentro del grid
                    in_grid = (0 <= cx < grid_max_x) and (0 <= cy < grid_max_y)
                    
                    # Asignación esperada
                    box_area = (x2 - x1) * (y2 - y1)
                    box_size = (box_area ** 0.5)
                    size_range = Config.FPN_SIZE_RANGES[level_idx]
                    should_assign = (box_area >= size_range[0]**2) and (box_area < size_range[1]**2)
                    
                    if should_assign:
                        status = "✓" if in_grid else "❌ OUT OF GRID"
                        print(f"      Box {box_idx}: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] " 
                              f"center=({cx:.0f},{cy:.0f}) {status}")
            else:
                print(f"    Image {batch_idx}: {num_pos} positive pixels")
            
            if num_pos > 0:
                labels = target['labels'][mask]
                print(f"      All labels are 1.0: {(labels == 1.0).all().item()}")
        
        print(f"    TOTAL positives at this level: {total_pos}")
    
    return loss_dict


def main():
    """Main debugging function"""
    
    print("\n" + "="*80)
    print("FCOS LOSS DEBUGGING".center(80))
    print("="*80)
    
    # Config
    config = Config()
    device = config.DEVICE
    
    print(f"\nDevice: {device}")
    print(f"Backbone frozen: {config.FREEZE_BACKBONE}")
    print(f"Box score threshold: {config.BOX_SCORE_THRESHOLD}")
    
    # Load data
    print("\n" + "="*80)
    print("Loading Data".center(80))
    print("="*80)
    
    train_loader, val_loader, test_loader, _ = create_dataloaders(config)
    print(f"\nTrain batches: {len(train_loader)}")
    
    # Build model
    print("\n" + "="*80)
    print("Building Model".center(80))
    print("="*80)
    
    model = build_model(config)
    model = model.to(device)
    model.train()
    
    # Loss
    criterion = FCOSLoss(config)
    
    # Analyze first batch
    print("\n" + "="*80)
    print("Analyzing First Batch".center(80))
    print("="*80)
    
    for batch_idx, (images, sketches, targets) in enumerate(train_loader):
        if batch_idx == 0:
            loss_dict = analyze_batch(images, sketches, targets, model, criterion, device)
            break
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY".center(80))
    print("="*80)
    
    num_pos = loss_dict.get('num_pos', 0)
    cls_loss = loss_dict['cls'].item() if torch.is_tensor(loss_dict['cls']) else loss_dict['cls']
    
    print(f"\n🔍 KEY INDICATORS:")
    print(f"  Positive pixels assigned: {num_pos}")
    print(f"  Classification loss: {cls_loss:.6f}")
    
    if num_pos == 0:
        print(f"\n❌ PROBLEM IDENTIFIED:")
        print(f"  NO positive targets are being assigned!")
        print(f"\n  Possible causes:")
        print(f"  1. BOX_SCORE_THRESHOLD too high ({config.BOX_SCORE_THRESHOLD})")
        print(f"     → All GT boxes filtered out before training")
        print(f"  2. FPN_SIZE_RANGES don't match object sizes")
        print(f"     → Objects fall outside all size ranges")
        print(f"  3. CENTER_SAMPLING_RADIUS too small")
        print(f"     → No pixels within sampling radius")
        print(f"\n  RECOMMENDED FIX:")
        print(f"  Set BOX_SCORE_THRESHOLD = 0.0 in config.py")
        
    elif cls_loss < 0.01:
        print(f"\n⚠️  PROBLEM IDENTIFIED:")
        print(f"  Classification loss is too low ({cls_loss:.6f})")
        print(f"  Expected: 0.3 - 0.8 for early training")
        print(f"\n  Possible causes:")
        print(f"  1. Model predicting all background")
        print(f"  2. Focal loss parameters too aggressive")
        print(f"  3. Imbalance between pos/neg samples")
        
    else:
        print(f"\n✅ Losses look reasonable")
        print(f"   But if mAP is still 0, check:")
        print(f"   1. Prediction decoding")
        print(f"   2. Evaluation metrics")
        print(f"   3. Learning rate / optimization")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()