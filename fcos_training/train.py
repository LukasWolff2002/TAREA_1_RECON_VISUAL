"""
Script principal de entrenamiento para Multi-Scale FCOS
"""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from data.dataset import create_dataloaders
from models.fcos_model import build_model
from models.loss import FCOSLoss
from fcos_utils.eval_utils import evaluate_model, AverageMeter


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, config, device):
    """
    Train for one epoch
    """
    model.train()
    
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    ctr_losses = AverageMeter()
    
    start_time = time.time()
    
    for iteration, (images, sketches, targets) in enumerate(train_loader):
        images = images.to(device)
        sketches = sketches.to(device)
        
        # Forward pass
        with autocast(enabled=config.USE_AMP):
            predictions = model(images, sketches)
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total']
            
            # Gradient accumulation
            loss = loss / config.ACCUMULATION_STEPS
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every ACCUMULATION_STEPS
        if (iteration + 1) % config.ACCUMULATION_STEPS == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics
        losses.update(loss_dict['total'].item(), images.size(0))
        # Verificamos si es un tensor antes de llamar a .item()
        cls_val = loss_dict['cls'].item() if torch.is_tensor(loss_dict['cls']) else loss_dict['cls']
        reg_val = loss_dict['reg'].item() if torch.is_tensor(loss_dict['reg']) else loss_dict['reg']
        ctr_val = loss_dict['ctr'].item() if torch.is_tensor(loss_dict['ctr']) else loss_dict['ctr']

        # Actualizamos los contadores de forma segura
        cls_losses.update(cls_val, images.size(0))
        reg_losses.update(reg_val, images.size(0))
        ctr_losses.update(ctr_val, images.size(0))
        
        # Print progress
        if (iteration + 1) % config.PRINT_FREQ == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
                  f"Iter [{iteration+1}/{len(train_loader)}] "
                  f"Loss: {losses.avg:.4f} (cls: {cls_losses.avg:.4f}, "
                  f"reg: {reg_losses.avg:.4f}, ctr: {ctr_losses.avg:.4f}) "
                  f"Time: {elapsed:.1f}s")
            start_time = time.time()
    
    return {
        'loss': losses.avg,
        'cls_loss': cls_losses.avg,
        'reg_loss': reg_losses.avg,
        'ctr_loss': ctr_losses.avg
    }


def main():
    # Configuration
    config = Config()
    config.display()
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Device
    device = config.DEVICE
    print(f"\nUsing device: {device}")
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    # Create dataloaders
    print("\n" + "="*70)
    print("Loading Dataset".center(70))
    print("="*70)
    train_loader, val_loader, test_loader, class_to_queries = create_dataloaders(config)
    
    print(f"\nDataset Statistics:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Classes: {len(class_to_queries)}")
    
    # Build model
    print("\n" + "="*70)
    print("Building Model".center(70))
    print("="*70)
    model = build_model(config)
    model = model.to(device)
    
    # Loss function
    criterion = FCOSLoss(config)
    
    # Optimizer (solo parámetros entrenables)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    print(f"\nOptimizer:")
    print(f"  Parameters being optimized: {sum(p.numel() for p in trainable_params):,}")
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        else:
            decay = 1.0
            for milestone in config.LR_DECAY_EPOCHS:
                if epoch >= milestone:
                    decay *= config.LR_DECAY_GAMMA
            return decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Resume from checkpoint if exists
    start_epoch = 1
    best_mAP = 0.0
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
    if os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('best_mAP', 0.0)
        print(f"Resumed from epoch {start_epoch-1}, best mAP: {best_mAP:.4f}")
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training".center(70))
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}".center(70))
        print(f"{'='*70}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, config, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate
        if epoch % config.EVAL_INTERVAL == 0 or epoch == config.NUM_EPOCHS:
            print("\nEvaluating on validation set...")
            val_metrics = evaluate_model(model, val_loader, config, device)
            
            print(f"\nValidation Results:")
            print(f"  mAP: {val_metrics['mAP']:.4f}")
            print(f"  Images: {val_metrics['num_images']}")
            
            # Save best model
            if val_metrics['mAP'] > best_mAP:
                best_mAP = val_metrics['mAP']
                best_path = os.path.join(config.CHECKPOINT_DIR, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_mAP': best_mAP,
                    'config': config.__dict__
                }, best_path)
                print(f"\n✓ Best model saved! mAP: {best_mAP:.4f}")
        
        # Save checkpoint
        if epoch % config.SAVE_INTERVAL == 0 or epoch == config.NUM_EPOCHS:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mAP': best_mAP,
                'config': config.__dict__
            }
            
            # Save latest
            latest_path = os.path.join(config.CHECKPOINT_DIR, 'latest.pth')
            torch.save(checkpoint, latest_path)
            
            # Save epoch checkpoint
            epoch_path = os.path.join(config.CHECKPOINT_DIR, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
            
            print(f"\n✓ Checkpoint saved: {epoch_path}")
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("Final Evaluation on Test Set".center(70))
    print("="*70 + "\n")
    
    # Load best model
    best_checkpoint = torch.load(
        os.path.join(config.CHECKPOINT_DIR, 'best.pth'),
        map_location=device
    )
    model.load_state_dict(best_checkpoint['model'])
    
    test_metrics = evaluate_model(model, test_loader, config, device)
    
    print(f"\nTest Results:")
    print(f"  mAP: {test_metrics['mAP']:.4f}")
    print(f"  Images: {test_metrics['num_images']}")
    
    print("\n" + "="*70)
    print("Training Complete!".center(70))
    print("="*70)
    print(f"\nBest Validation mAP: {best_mAP:.4f}")
    print(f"Test mAP: {test_metrics['mAP']:.4f}")


if __name__ == "__main__":
    main()