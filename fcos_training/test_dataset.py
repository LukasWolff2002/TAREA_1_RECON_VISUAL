"""
Script de prueba rápida para verificar que el dataset se carga correctamente
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from data.dataset import create_dataloaders
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_batch(images, sketches, targets, num_samples=2):
    """
    Visualizar un batch del dataset
    """
    batch_size = min(num_samples, len(images))
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax = axes[i, 0]
        ax.imshow(img)
        
        # Draw boxes
        boxes = targets[i]['boxes'].cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        
        ax.set_title(f'Image (Class ID: {targets[i]["class_id"]})\n{len(boxes)} boxes')
        ax.axis('off')
        
        # Sketch
        sketch = sketches[i].permute(1, 2, 0).cpu().numpy()
        sketch = std * sketch + mean
        sketch = np.clip(sketch, 0, 1)
        
        axes[i, 1].imshow(sketch)
        axes[i, 1].set_title('Query Sketch')
        axes[i, 1].axis('off')
        
        # Image with zoom on first box
        if len(boxes) > 0:
            ax = axes[i, 2]
            ax.imshow(img)
            
            # Zoom to first box with some margin
            box = boxes[0]
            x1, y1, x2, y2 = box
            margin = 50
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img.shape[1], x2 + margin)
            y2 = min(img.shape[0], y2 + margin)
            
            ax.set_xlim(x1, x2)
            ax.set_ylim(y2, y1)  # Inverted y-axis
            
            # Draw box
            rect = patches.Rectangle(
                (boxes[0][0], boxes[0][1]), 
                boxes[0][2] - boxes[0][0], 
                boxes[0][3] - boxes[0][1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title('Zoomed First Box')
            ax.axis('off')
        else:
            axes[i, 2].text(0.5, 0.5, 'No boxes', 
                          ha='center', va='center', fontsize=14)
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to dataset_sample.png")
    plt.show()


def main():
    print("="*70)
    print("DATASET LOADING TEST".center(70))
    print("="*70)
    
    # Config
    config = Config()
    print(f"\nUsing dataset: {config.DATASET_JSON}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Train size: {config.TRAIN_SIZES}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, class_to_queries = create_dataloaders(config)
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Number of classes: {len(class_to_queries)}")
    
    # Show class info
    print(f"\nClass Information:")
    for class_id, info in list(class_to_queries.items())[:5]:
        print(f"  [{class_id:2d}] {info['name']:15s} - {len(info['queries'])} queries")
    if len(class_to_queries) > 5:
        print(f"  ... and {len(class_to_queries) - 5} more classes")
    
    # Get a batch
    print("\nFetching a sample batch...")
    images, sketches, targets = next(iter(train_loader))
    
    print(f"\nBatch info:")
    print(f"  Images shape: {images.shape}")
    print(f"  Sketches shape: {sketches.shape}")
    print(f"  Number of samples: {len(targets)}")
    
    print(f"\nSample details:")
    for i, target in enumerate(targets):
        print(f"  Sample {i}:")
        print(f"    Class ID: {target['class_id']}")
        print(f"    Number of boxes: {len(target['boxes'])}")
        print(f"    Original size: {target['orig_size'].tolist()}")
        print(f"    Resized size: {target['size'].tolist()}")
        print(f"    Scale factor: {target['scale']:.3f}")
    
    # Visualize
    print("\nVisualizing samples...")
    visualize_batch(images, sketches, targets, num_samples=2)
    
    print("\n" + "="*70)
    print("TEST COMPLETE!".center(70))
    print("="*70)
    print("\n✓ Dataset is ready for training!")
    print("  Run 'python train.py' to start training")


if __name__ == "__main__":
    main()