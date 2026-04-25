"""
Script de inferencia mejorado: Retorna siempre la MEJOR detección (Top-1 score)
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.fcos_model import build_model
from fcos_utils.eval_utils import decode_predictions


def load_trained_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    model = build_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint.get('epoch', 'Unknown')} epochs")
    print(f"Best mAP: {checkpoint.get('best_mAP', 'N/A')}")
    
    return model


def preprocess_image(image_path, target_size=1024):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    image_resized = image.resize((new_w, new_h), Image.BILINEAR)
    
    padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    padded_image.paste(image_resized, (0, 0))
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image_tensor = transforms.ToTensor()(padded_image)
    image_tensor = normalize(image_tensor)
    
    return image_tensor.unsqueeze(0), image, scale


def preprocess_sketch(sketch_path, sketch_size=224):
    """Preprocess sketch query"""
    sketch = Image.open(sketch_path).convert('RGB')
    sketch = sketch.resize((sketch_size, sketch_size))
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    sketch_tensor = transforms.ToTensor()(sketch)
    sketch_tensor = normalize(sketch_tensor)
    
    return sketch_tensor.unsqueeze(0), sketch


def visualize_best_detection(image, sketch, best_box, best_score, save_path=None):
    """Visualize ONLY the best detection result"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image with the best detection
    ax = axes[0]
    ax.imshow(image)
    
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3,  # Made slightly thicker to highlight it's the best one
            edgecolor='#00FF00', # Green for the top prediction
            facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(
            x1, y1 - 5,
            f'Top Match: {best_score:.3f}',
            color='#00FF00',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
        )
        ax.set_title(f'Top Prediction (Score: {best_score:.3f})', fontsize=14)
    else:
        ax.set_title('No objects detected at all', fontsize=14, color='red')
        
    ax.axis('off')
    
    # Query sketch
    axes[1].imshow(sketch)
    axes[1].set_title('Query Sketch', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    # Use block=False if running in certain remote environments to prevent hanging
    plt.show()


def run_inference(model, image_path, sketch_path, config, device, save_path=None):
    """Run inference and extract the single best bounding box"""
    image_tensor, image_orig, scale = preprocess_image(image_path, config.TEST_SIZE)
    sketch_tensor, sketch_orig = preprocess_sketch(sketch_path, config.SKETCH_SIZE)
    
    image_tensor = image_tensor.to(device)
    sketch_tensor = sketch_tensor.to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor, sketch_tensor)
        detections = decode_predictions(predictions, config)
    
    boxes = detections[0]['boxes'].cpu().numpy()
    scores = detections[0]['scores'].cpu().numpy()
    
    best_box = None
    best_score = 0.0

    print(f"\nDetection Results:")
    if len(boxes) > 0:
        # The decode_predictions function (if standard) might already sort by score, 
        # but let's be explicit and find the argmax just to be safe.
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        # Scale the best box back to original image size
        best_box = boxes[best_idx] / scale
        
        print(f"  Found {len(boxes)} candidates. Selecting the top match.")
        print(f"  Best Confidence Score: {best_score:.4f}")
        print(f"  Best Bounding Box: {best_box}")
    else:
        print("  Model did not return any predictions even before NMS.")

    visualize_best_detection(image_orig, sketch_orig, best_box, best_score, save_path)
    
    return best_box, best_score


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FCOS inference (Top-1)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--sketch', type=str, required=True,
                        help='Path to query sketch')
    parser.add_argument('--output', type=str, default='best_detection_result.png',
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Config
    config = Config()
    # Force the threshold to be extremely low so the model outputs ALL possible candidates.
    # We will sort through them and pick the top 1 regardless of absolute score.
    config.SCORE_THRESHOLD = 0.001 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model = load_trained_model(args.checkpoint, config, device)
    
    run_inference(
        model, args.image, args.sketch, config, device, args.output
    )


if __name__ == "__main__":
    main()