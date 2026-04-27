"""
Script de Evaluación Completa para FCOS
Genera métricas cuantitativas y visualizaciones

CONFIGURACIÓN: Modifica las variables en la sección CONFIGURATION
"""
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcos_training.configs.config import Config
from fcos_training.data.dataset import create_dataloaders
from fcos_training.models.fcos_model import build_model
from fcos_training.fcos_utils.eval_utils import decode_predictions


# ============================================================================
# CONFIGURATION - MODIFICA ESTAS VARIABLES
# ============================================================================
CHECKPOINT_PATH = "fcos_training/checkpoints/best.pth"
OUTPUT_DIR = "./evaluation_results"
DATASET_SPLIT = "test"  # Opciones: "train", "val", "test"
# ============================================================================


class FCOSEvaluator:
    """
    Evaluador completo para FCOS
    """
    def __init__(self, model, dataloader, config, device):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = device
        
        self.results = {
            'predictions': [],
            'targets': [],
            'metrics': {},
            'timing': [],
            'per_level_stats': defaultdict(list)
        }
    
    def compute_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)
        
        return iou
    
    def compute_ap(self, pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
        """Compute Average Precision"""
        if len(gt_boxes) == 0:
            return 1.0 if len(pred_boxes) == 0 else 0.0
        
        if len(pred_boxes) == 0:
            return 0.0
        
        # Sort by score
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        # Compute IoU matrix
        ious = self.compute_iou(pred_boxes, gt_boxes)
        
        # Match predictions to GT
        matched = np.zeros(len(gt_boxes), dtype=bool)
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        
        for i in range(len(pred_boxes)):
            max_iou = 0
            max_idx = -1
            
            for j in range(len(gt_boxes)):
                if not matched[j] and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    max_idx = j
            
            if max_iou >= iou_threshold:
                tp[i] = 1
                matched[max_idx] = True
            else:
                fp[i] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP (11-point interpolation)
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap, precisions, recalls, tp, fp
    
    def assign_to_fpn_level(self, boxes):
        """Assign boxes to FPN levels based on size"""
        if len(boxes) == 0:
            return []
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sizes = np.sqrt(areas)
        
        levels = []
        for size in sizes:
            level = -1
            for idx, (min_size, max_size) in enumerate(self.config.FPN_SIZE_RANGES):
                if min_size <= size < max_size:
                    level = idx
                    break
            levels.append(level)
        
        return levels
    
    def run_inference(self):
        """Run inference on entire dataset"""
        self.model.eval()
        
        print("\n" + "="*80)
        print("RUNNING INFERENCE".center(80))
        print("="*80 + "\n")
        
        with torch.no_grad():
            for batch_idx, (images, sketches, targets) in enumerate(tqdm(
                self.dataloader, desc="Processing batches"
            )):
                images = images.to(self.device)
                sketches = sketches.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                predictions = self.model(images, sketches)
                detections = decode_predictions(predictions, self.config)
                inference_time = time.time() - start_time
                
                # Store results
                for det, target in zip(detections, targets):
                    pred_boxes = det['boxes'].cpu().numpy()
                    pred_scores = det['scores'].cpu().numpy()
                    gt_boxes = target['boxes'].cpu().numpy()
                    
                    self.results['predictions'].append({
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'image_id': target['image_id']
                    })
                    
                    self.results['targets'].append({
                        'boxes': gt_boxes,
                        'image_id': target['image_id']
                    })
                    
                    self.results['timing'].append(inference_time / len(targets))
                    
                    # Per-level statistics
                    if len(gt_boxes) > 0:
                        gt_levels = self.assign_to_fpn_level(gt_boxes)
                        for level in gt_levels:
                            if level >= 0:
                                self.results['per_level_stats'][f'P{level+3}'].append(1)
        
        print(f"\n✓ Processed {len(self.results['predictions'])} images")
        print(f"  Average inference time: {np.mean(self.results['timing'])*1000:.2f} ms/image")
    
    def compute_metrics(self):
        """Compute all evaluation metrics"""
        print("\n" + "="*80)
        print("COMPUTING METRICS".center(80))
        print("="*80 + "\n")
        
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        # Compute mAP for different IoU thresholds
        maps = {}
        all_precisions = defaultdict(list)
        all_recalls = defaultdict(list)
        all_tp = defaultdict(list)
        all_fp = defaultdict(list)
        
        for iou_thresh in tqdm(iou_thresholds, desc="Computing mAP"):
            aps = []
            for pred, target in zip(self.results['predictions'], self.results['targets']):
                ap, prec, rec, tp, fp = self.compute_ap(
                    pred['boxes'], 
                    pred['scores'], 
                    target['boxes'],
                    iou_threshold=iou_thresh
                )
                aps.append(ap)
                all_precisions[iou_thresh].append(prec)
                all_recalls[iou_thresh].append(rec)
                all_tp[iou_thresh].extend(tp)
                all_fp[iou_thresh].extend(fp)
            
            maps[f'mAP@{iou_thresh:.2f}'] = np.mean(aps)
        
        # COCO-style mAP (average over IoU thresholds)
        maps['mAP@[0.5:0.95]'] = np.mean(list(maps.values()))
        
        # Detection statistics
        num_detections = [len(p['boxes']) for p in self.results['predictions']]
        num_gt = [len(t['boxes']) for t in self.results['targets']]
        
        # Score distribution
        all_scores = np.concatenate([p['scores'] for p in self.results['predictions'] if len(p['scores']) > 0])
        
        # Store metrics
        self.results['metrics'] = {
            'mAP': maps,
            'detection_stats': {
                'mean_detections_per_image': np.mean(num_detections),
                'std_detections_per_image': np.std(num_detections),
                'max_detections_per_image': np.max(num_detections),
                'mean_gt_per_image': np.mean(num_gt),
                'total_predictions': sum(num_detections),
                'total_gt': sum(num_gt)
            },
            'score_stats': {
                'mean': float(np.mean(all_scores)) if len(all_scores) > 0 else 0,
                'std': float(np.std(all_scores)) if len(all_scores) > 0 else 0,
                'min': float(np.min(all_scores)) if len(all_scores) > 0 else 0,
                'max': float(np.max(all_scores)) if len(all_scores) > 0 else 0,
                'median': float(np.median(all_scores)) if len(all_scores) > 0 else 0,
            },
            'timing': {
                'mean_ms': np.mean(self.results['timing']) * 1000,
                'std_ms': np.std(self.results['timing']) * 1000,
                'fps': 1.0 / np.mean(self.results['timing'])
            },
            'precision_recall': {
                str(iou): {
                    'precisions': [p.tolist() for p in all_precisions[iou]],
                    'recalls': [r.tolist() for r in all_recalls[iou]]
                } for iou in [0.5, 0.75]  # Save only main thresholds
            }
        }
        
        # Per-level statistics
        for level, counts in self.results['per_level_stats'].items():
            self.results['metrics'][f'{level}_gt_count'] = sum(counts)
        
        # Print summary
        print("\n📊 METRICS SUMMARY")
        print("-" * 80)
        for metric, value in self.results['metrics']['mAP'].items():
            print(f"  {metric:<20} {value:.4f}")
        
        print(f"\n📈 DETECTION STATISTICS")
        print("-" * 80)
        for key, value in self.results['metrics']['detection_stats'].items():
            print(f"  {key:<30} {value:.2f}")
        
        print(f"\n⚡ TIMING")
        print("-" * 80)
        print(f"  Mean inference time:  {self.results['metrics']['timing']['mean_ms']:.2f} ms")
        print(f"  Throughput:           {self.results['metrics']['timing']['fps']:.2f} FPS")
    
    def generate_visualizations(self, output_dir):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS".center(80))
        print("="*80 + "\n")
        
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. mAP bar chart
        self._plot_map_comparison(vis_dir)
        
        # 2. Precision-Recall curves
        self._plot_pr_curves(vis_dir)
        
        # 3. Score distribution
        self._plot_score_distribution(vis_dir)
        
        # 4. Detection count distribution
        self._plot_detection_distribution(vis_dir)
        
        # 5. FPN level distribution
        self._plot_fpn_distribution(vis_dir)
        
        # 6. Size vs AP scatter
        self._plot_size_vs_ap(vis_dir)
        
        print(f"\n✓ Saved visualizations to {vis_dir}")
    
    def _plot_map_comparison(self, output_dir):
        """Plot mAP for different IoU thresholds"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = self.results['metrics']['mAP']
        thresholds = [k for k in metrics.keys() if k.startswith('mAP@0')]
        values = [metrics[k] for k in thresholds]
        
        # Clean labels
        labels = [k.replace('mAP@', '') for k in thresholds]
        
        bars = ax.bar(labels, values, color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.axhline(y=metrics['mAP@[0.5:0.95]'], color='red', linestyle='--', 
                   label=f'mAP@[0.5:0.95] = {metrics["mAP@[0.5:0.95]"]:.3f}')
        
        ax.set_xlabel('IoU Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP', fontsize=12, fontweight='bold')
        ax.set_title('Mean Average Precision at Different IoU Thresholds', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'map_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self, output_dir):
        """Plot Precision-Recall curves"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, iou_thresh in enumerate([0.5, 0.75]):
            ax = axes[idx]
            
            pr_data = self.results['metrics']['precision_recall'][str(iou_thresh)]
            
            # Average precision-recall across all images
            max_len = max(len(r) for r in pr_data['recalls'] if len(r) > 0)
            
            if max_len > 0:
                # Interpolate to common recall points
                recall_points = np.linspace(0, 1, 100)
                precisions_interp = []
                
                for prec_list, rec_list in zip(pr_data['precisions'], pr_data['recalls']):
                    if len(rec_list) > 0:
                        prec_interp = np.interp(recall_points, rec_list, prec_list, left=0, right=0)
                        precisions_interp.append(prec_interp)
                
                if precisions_interp:
                    mean_precision = np.mean(precisions_interp, axis=0)
                    std_precision = np.std(precisions_interp, axis=0)
                    
                    ax.plot(recall_points, mean_precision, 'b-', linewidth=2, label='Mean P-R curve')
                    ax.fill_between(recall_points, 
                                   mean_precision - std_precision,
                                   mean_precision + std_precision,
                                   alpha=0.2, color='blue', label='±1 std')
            
            ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
            ax.set_title(f'Precision-Recall Curve @ IoU {iou_thresh}', 
                        fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distribution(self, output_dir):
        """Plot confidence score distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_scores = np.concatenate([
            p['scores'] for p in self.results['predictions'] if len(p['scores']) > 0
        ])
        
        ax.hist(all_scores, bins=50, color='coral', alpha=0.7, edgecolor='black')
        
        # Add mean and median lines
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_score:.3f}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2,
                  label=f'Median = {median_score:.3f}')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Detection Confidence Scores', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detection_distribution(self, output_dir):
        """Plot distribution of detections per image"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        num_pred = [len(p['boxes']) for p in self.results['predictions']]
        num_gt = [len(t['boxes']) for t in self.results['targets']]
        
        # Predictions
        ax = axes[0]
        ax.hist(num_pred, bins=range(0, max(num_pred)+2), color='steelblue', 
               alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(num_pred), color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {np.mean(num_pred):.2f}')
        ax.set_xlabel('Number of Predictions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Predictions per Image', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ground Truth
        ax = axes[1]
        ax.hist(num_gt, bins=range(0, max(num_gt)+2), color='forestgreen', 
               alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(num_gt), color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {np.mean(num_gt):.2f}')
        ax.set_xlabel('Number of Ground Truth Boxes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Ground Truth Boxes per Image', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fpn_distribution(self, output_dir):
        """Plot FPN level distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        levels = ['P3', 'P4', 'P5', 'P6']
        counts = [
            self.results['metrics'].get(f'{level}_gt_count', 0) 
            for level in levels
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(levels, counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add size ranges as annotations
        size_ranges = self.config.FPN_SIZE_RANGES
        strides = self.config.FPN_STRIDES
        
        for i, (level, size_range, stride) in enumerate(zip(levels, size_ranges, strides)):
            if size_range[1] == 999999:
                range_str = f'[{size_range[0]}+ px]\nStride: {stride}'
            else:
                range_str = f'[{size_range[0]}-{size_range[1]} px]\nStride: {stride}'
            
            ax.text(i, -max(counts)*0.15, range_str, 
                   ha='center', va='top', fontsize=9, style='italic')
        
        ax.set_ylabel('Number of Ground Truth Boxes', fontsize=12, fontweight='bold')
        ax.set_title('FPN Level Assignment Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fpn_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_size_vs_ap(self, output_dir):
        """Plot object size vs AP"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compute AP for each image and get average GT box size
        aps = []
        avg_sizes = []
        
        for pred, target in zip(self.results['predictions'], self.results['targets']):
            if len(target['boxes']) > 0:
                ap, _, _, _, _ = self.compute_ap(
                    pred['boxes'], pred['scores'], target['boxes'], iou_threshold=0.5
                )
                aps.append(ap)
                
                # Average box size
                boxes = target['boxes']
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                avg_sizes.append(np.sqrt(np.mean(areas)))
        
        if len(aps) > 0:
            scatter = ax.scatter(avg_sizes, aps, c=aps, cmap='RdYlGn', 
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(avg_sizes, aps, 1)
            p = np.poly1d(z)
            ax.plot(sorted(avg_sizes), p(sorted(avg_sizes)), "r--", alpha=0.8, linewidth=2,
                   label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
            
            plt.colorbar(scatter, ax=ax, label='AP@0.5')
            ax.set_xlabel('Average Object Size (px)', fontsize=12, fontweight='bold')
            ax.set_ylabel('AP@0.5', fontsize=12, fontweight='bold')
            ax.set_title('Object Size vs Average Precision', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'size_vs_ap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary_report(self):
        """Print text summary report to console"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY REPORT".center(80))
        print("="*80 + "\n")
        
        # Main metrics
        print("📊 MAIN METRICS")
        print("-" * 80)
        print(f"  mAP @ IoU 0.5:     {self.results['metrics']['mAP']['mAP@0.50']:.4f}")
        print(f"  mAP @ IoU 0.75:    {self.results['metrics']['mAP']['mAP@0.75']:.4f}")
        print(f"  mAP [0.5:0.95]:    {self.results['metrics']['mAP']['mAP@[0.5:0.95]']:.4f}")
        
        # Detection stats
        print(f"\n📈 DETECTION STATISTICS")
        print("-" * 80)
        stats = self.results['metrics']['detection_stats']
        print(f"  Total Predictions:        {stats['total_predictions']:.0f}")
        print(f"  Total Ground Truth:       {stats['total_gt']:.0f}")
        print(f"  Mean Detections/Image:    {stats['mean_detections_per_image']:.2f} ± {stats['std_detections_per_image']:.2f}")
        print(f"  Max Detections/Image:     {stats['max_detections_per_image']:.0f}")
        
        # Timing
        print(f"\n⚡ PERFORMANCE")
        print("-" * 80)
        timing = self.results['metrics']['timing']
        print(f"  Inference Time:           {timing['mean_ms']:.2f} ms/image")
        print(f"  Throughput:               {timing['fps']:.2f} FPS")
        
        # Score stats
        print(f"\n⭐ CONFIDENCE SCORES")
        print("-" * 80)
        scores = self.results['metrics']['score_stats']
        print(f"  Mean:    {scores['mean']:.3f}")
        print(f"  Median:  {scores['median']:.3f}")
        print(f"  Std:     {scores['std']:.3f}")
        print(f"  Range:   [{scores['min']:.3f}, {scores['max']:.3f}]")
        
        print("\n" + "="*80)
    
    def save_results(self, output_dir):
        """Save results to JSON"""
        results_path = Path(output_dir) / 'evaluation_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'metrics': self.results['metrics'],
            'num_images': len(self.results['predictions']),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Saved results to: {results_path}")


def main():
    """Main evaluation function"""
    # Configuration
    config = Config()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Device
    device = config.DEVICE
    
    print("\n" + "="*80)
    print("FCOS MODEL EVALUATION".center(80))
    print("="*80)
    print(f"\n📁 Checkpoint:  {CHECKPOINT_PATH}")
    print(f"📂 Output:      {OUTPUT_DIR}")
    print(f"📊 Split:       {DATASET_SPLIT}")
    print(f"🖥️  Device:      {device}\n")
    
    # Load model
    print("="*80)
    print("LOADING MODEL".center(80))
    print("="*80 + "\n")
    
    model = build_model(config)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded checkpoint from: {CHECKPOINT_PATH}")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Best mAP: {checkpoint.get('best_mAP', 'N/A')}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATASET".center(80))
    print("="*80 + "\n")
    
    train_loader, val_loader, test_loader, _ = create_dataloaders(config)
    
    # Select dataloader
    if DATASET_SPLIT == 'train':
        dataloader = train_loader
    elif DATASET_SPLIT == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader
    
    print(f"✓ Loaded {DATASET_SPLIT} set: {len(dataloader)} batches")
    
    # Create evaluator
    evaluator = FCOSEvaluator(model, dataloader, config, device)
    
    # Run evaluation
    evaluator.run_inference()
    evaluator.compute_metrics()
    evaluator.generate_visualizations(output_dir)
    evaluator.print_summary_report()
    evaluator.save_results(output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!".center(80))
    print("="*80)
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Visualizations: {output_dir / 'visualizations'}")
    print(f"✓ Metrics JSON:   {output_dir / 'evaluation_results.json'}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()