"""
Script para analizar el dataset antes de entrenar
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config


def analyze_dataset(json_path):
    """
    Analizar dataset y generar estadísticas
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("DATASET ANALYSIS".center(80))
    print("="*80)
    
    # Summary
    summary = data['summary']['counts']
    print("\n📊 SUMMARY")
    print("-"*80)
    for key, value in summary.items():
        print(f"  {key:.<50} {value:>10,}")
    
    # Class distribution
    print("\n📦 CLASS DISTRIBUTION")
    print("-"*80)
    class_samples = defaultdict(int)
    class_boxes = defaultdict(int)
    class_queries = {}
    
    for cls in data['classes']:
        class_id = cls['class_id']
        class_name = cls['class_name']
        class_queries[class_id] = (class_name, len(cls['queries']))
    
    for sample in data['samples']:
        class_id = sample['class_id']
        class_samples[class_id] += 1
        class_boxes[class_id] += len(sample['boxes'])
    
    # Sort by number of samples
    sorted_classes = sorted(class_queries.items(), 
                           key=lambda x: class_samples[x[0]], 
                           reverse=True)
    
    print(f"\n{'Class':<15} {'Samples':>10} {'Boxes':>10} {'Queries':>10} {'Boxes/Sample':>15}")
    print("-"*80)
    for class_id, (class_name, n_queries) in sorted_classes:
        n_samples = class_samples[class_id]
        n_boxes = class_boxes[class_id]
        boxes_per_sample = n_boxes / n_samples if n_samples > 0 else 0
        print(f"{class_name:<15} {n_samples:>10,} {n_boxes:>10,} {n_queries:>10,} {boxes_per_sample:>15.2f}")
    
    # Box size distribution
    print("\n📏 BOUNDING BOX SIZE DISTRIBUTION")
    print("-"*80)
    
    box_widths = []
    box_heights = []
    box_areas = []
    
    for sample in data['samples']:
        for box in sample['boxes']:
            x1, y1, x2, y2 = box['bbox_xyxy']
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            box_widths.append(w)
            box_heights.append(h)
            box_areas.append(area)
    
    box_widths = np.array(box_widths)
    box_heights = np.array(box_heights)
    box_areas = np.array(box_areas)
    box_sizes = np.sqrt(box_areas)  # Characteristic size
    
    print(f"\nWidth Statistics:")
    print(f"  Min: {box_widths.min():.1f} px")
    print(f"  Max: {box_widths.max():.1f} px")
    print(f"  Mean: {box_widths.mean():.1f} px")
    print(f"  Median: {np.median(box_widths):.1f} px")
    print(f"  Std: {box_widths.std():.1f} px")
    
    print(f"\nHeight Statistics:")
    print(f"  Min: {box_heights.min():.1f} px")
    print(f"  Max: {box_heights.max():.1f} px")
    print(f"  Mean: {box_heights.mean():.1f} px")
    print(f"  Median: {np.median(box_heights):.1f} px")
    print(f"  Std: {box_heights.std():.1f} px")
    
    print(f"\nArea Statistics:")
    print(f"  Min: {box_areas.min():.1f} px²")
    print(f"  Max: {box_areas.max():.1f} px²")
    print(f"  Mean: {box_areas.mean():.1f} px²")
    print(f"  Median: {np.median(box_areas):.1f} px²")
    
    # FPN level assignment
    config = Config()
    level_counts = {i: 0 for i in range(len(config.FPN_SIZE_RANGES))}
    
    for size in box_sizes:
        for level, (min_size, max_size) in enumerate(config.FPN_SIZE_RANGES):
            if min_size <= size < max_size:
                level_counts[level] += 1
                break
    
    print(f"\n🎯 FPN LEVEL ASSIGNMENT (based on sqrt(area))")
    print("-"*80)
    total = sum(level_counts.values())
    for level, count in level_counts.items():
        pct = 100 * count / total if total > 0 else 0
        size_range = config.FPN_SIZE_RANGES[level]
        stride = config.FPN_STRIDES[level]
        print(f"  P{level+3} (stride {stride:2d}): {count:>6,} boxes ({pct:>5.1f}%)  "
              f"[{size_range[0]:>3d}-{size_range[1] if size_range[1] < 999999 else 'inf':>3s} px]")
    
    # Image size distribution
    print(f"\n🖼️  IMAGE SIZE DISTRIBUTION")
    print("-"*80)
    
    image_widths = []
    image_heights = []
    
    for sample in data['samples']:
        image_widths.append(sample['page_width'])
        image_heights.append(sample['page_height'])
    
    image_widths = np.array(image_widths)
    image_heights = np.array(image_heights)
    
    print(f"Width range: {image_widths.min():.0f} - {image_widths.max():.0f} px")
    print(f"Height range: {image_heights.min():.0f} - {image_heights.max():.0f} px")
    print(f"Mean size: {image_widths.mean():.0f} × {image_heights.mean():.0f} px")
    
    aspect_ratios = image_widths / image_heights
    print(f"\nAspect ratio range: {aspect_ratios.min():.2f} - {aspect_ratios.max():.2f}")
    print(f"Mean aspect ratio: {aspect_ratios.mean():.2f}")
    
    # Score distribution
    print(f"\n⭐ ANNOTATION SCORE DISTRIBUTION")
    print("-"*80)
    
    scores = []
    for sample in data['samples']:
        for box in sample['boxes']:
            scores.append(box['score'])
    
    scores = np.array(scores)
    
    print(f"Score range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Median score: {np.median(scores):.3f}")
    
    # Score percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\nScore Percentiles:")
    for p in percentiles:
        val = np.percentile(scores, p)
        print(f"  {p}th: {val:.3f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-"*80)
    
    # Class imbalance
    max_samples = max(class_samples.values())
    min_samples = min(class_samples.values())
    imbalance_ratio = max_samples / min_samples
    
    print(f"\n1. Class Imbalance:")
    print(f"   Ratio: {imbalance_ratio:.1f}x (max: {max_samples}, min: {min_samples})")
    if imbalance_ratio > 10:
        print(f"   ⚠️  Severe imbalance! Enable class balancing (USE_CLASS_BALANCING=True)")
    else:
        print(f"   ✓ Moderate imbalance, class balancing recommended")
    
    # Score threshold
    score_threshold = np.percentile(scores, 30)
    print(f"\n2. Box Filtering:")
    print(f"   30th percentile score: {score_threshold:.3f}")
    print(f"   Recommended BOX_SCORE_THRESHOLD: {max(0.6, score_threshold):.2f}")
    
    # Image sizes
    print(f"\n3. Image Resizing:")
    print(f"   Max image: {image_widths.max():.0f} × {image_heights.max():.0f}")
    print(f"   Recommended MAX_SIZE: {max(1600, int(max(image_widths.max(), image_heights.max()) * 1.1))}")
    
    # Visualizations
    print(f"\n📊 Generating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Class distribution
    ax = axes[0, 0]
    class_names = [name for _, (name, _) in sorted_classes[:10]]
    sample_counts = [class_samples[cid] for cid, _ in sorted_classes[:10]]
    ax.barh(class_names, sample_counts)
    ax.set_xlabel('Number of Samples')
    ax.set_title('Top 10 Classes by Sample Count')
    ax.invert_yaxis()
    
    # 2. Box size distribution
    ax = axes[0, 1]
    ax.hist(box_sizes, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Box Size (sqrt(area))')
    ax.set_ylabel('Count')
    ax.set_title('Bounding Box Size Distribution')
    ax.axvline(64, color='r', linestyle='--', label='P3/P4 boundary')
    ax.axvline(128, color='g', linestyle='--', label='P4/P5 boundary')
    ax.axvline(256, color='b', linestyle='--', label='P5/P6 boundary')
    ax.legend()
    
    # 3. Score distribution
    ax = axes[0, 2]
    ax.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Annotation Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution')
    ax.axvline(0.6, color='r', linestyle='--', label='Threshold=0.6')
    ax.legend()
    
    # 4. Aspect ratio vs size
    ax = axes[1, 0]
    scatter = ax.scatter(box_widths, box_heights, 
                        c=scores, cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Width (px)')
    ax.set_ylabel('Height (px)')
    ax.set_title('Box Width vs Height (colored by score)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.colorbar(scatter, ax=ax)
    
    # 5. FPN level distribution
    ax = axes[1, 1]
    levels = [f'P{i+3}' for i in range(4)]
    counts = [level_counts[i] for i in range(4)]
    ax.bar(levels, counts)
    ax.set_ylabel('Number of Boxes')
    ax.set_title('FPN Level Assignment')
    
    # 6. Image size distribution
    ax = axes[1, 2]
    ax.scatter(image_widths, image_heights, alpha=0.5, s=20)
    ax.set_xlabel('Width (px)')
    ax.set_ylabel('Height (px)')
    ax.set_title('Image Size Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to dataset_analysis.png")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE".center(80))
    print("="*80)


if __name__ == "__main__":
    config = Config()
    analyze_dataset(config.DATASET_JSON)