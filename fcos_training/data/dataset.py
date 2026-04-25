"""
Dataset loader para Query-Conditioned Detection con sketches
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2


class QueryDetectionDataset(Dataset):
    """
    Dataset para detección condicionada por query usando sketches
    """
    def __init__(self, samples, class_to_queries, config, mode='train'):
        """
        Args:
            samples: Lista de samples (imagen + boxes + clase)
            class_to_queries: Dict mapping class_id -> queries
            config: Config object
            mode: 'train', 'val', or 'test'
        """
        self.samples = samples
        self.class_to_queries = class_to_queries
        self.config = config
        self.mode = mode
        
        # Normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Augmentation
        if mode == 'train' and config.USE_AUGMENTATION:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        else:
            self.color_jitter = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = sample['image_path']
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Load boxes (already filtered by score)
        boxes = []
        for box_dict in sample['boxes']:
            # Use xyxy format
            x1, y1, x2, y2 = box_dict['bbox_xyxy']
            boxes.append([x1, y1, x2, y2])
        
        boxes = np.array(boxes, dtype=np.float32)
        
        # Randomly select a query sketch for this class
        queries = sample['queries']
        query_info = random.choice(queries)
        query_path = query_info['query_path']
        
        # Load sketch
        sketch = Image.open(query_path).convert('RGB')
        sketch = sketch.resize((self.config.SKETCH_SIZE, self.config.SKETCH_SIZE))
        
        # Apply augmentations
        if self.mode == 'train':
            image, boxes = self._apply_augmentations(image, boxes)
        
        # Resize image with padding
        image, boxes, scale, pad_w, pad_h = self._resize_with_padding(
            image, boxes, self.mode
        )
        
        # Convert to tensor
        image_tensor = transforms.ToTensor()(image)
        sketch_tensor = transforms.ToTensor()(sketch)
        
        # Apply color jitter to image only (not sketch)
        if self.color_jitter is not None:
            image_tensor = self.color_jitter(image_tensor)
        
        # Normalize
        image_tensor = self.normalize(image_tensor)
        sketch_tensor = self.normalize(sketch_tensor)
        
        # Prepare target
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'class_id': sample['class_id'],
            'image_id': sample['sample_id'],
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor(image_tensor.shape[-2:]),
            'scale': scale,
            'pad': torch.tensor([pad_w, pad_h])
        }
        
        return image_tensor, sketch_tensor, target
    
    def _apply_augmentations(self, image, boxes):
        """Apply random augmentations"""
        # Random horizontal flip
        if random.random() < self.config.HORIZONTAL_FLIP_PROB:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w = image.size[0]
            boxes_flipped = boxes.copy()
            boxes_flipped[:, [0, 2]] = w - boxes[:, [2, 0]]
            boxes = boxes_flipped
        
        return image, boxes
    
    def _resize_with_padding(self, image, boxes, mode):
        """
        Resize image maintaining aspect ratio and pad to square
        """
        orig_w, orig_h = image.size
        
        # Determine target size
        if mode == 'train':
            target_size = random.choice(self.config.TRAIN_SIZES)
        else:
            target_size = self.config.TEST_SIZE
        
        # Calculate scale
        scale = min(target_size / orig_w, target_size / orig_h)
        scale = min(scale, self.config.MAX_SIZE / max(orig_w, orig_h))
        
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize
        image_resized = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Pad to square
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        
        # Create padded image
        padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        padded_image.paste(image_resized, (0, 0))
        
        # Scale boxes
        boxes_scaled = boxes * scale
        
        return padded_image, boxes_scaled, scale, pad_w, pad_h
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function para procesar lotes con imágenes de diferentes tamaños.
        Aplica zero-padding a la derecha y abajo para igualar el tamaño máximo del batch.
        """
        images = [item[0] for item in batch]
        sketches = [item[1] for item in batch]
        targets = [item[2] for item in batch]
        
        # 1. Encontrar el alto y ancho máximo en este batch específico
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        
        # 2. Rellenar (pad) las imágenes para que todas tengan (max_h, max_w)
        padded_images = []
        for img in images:
            # F.pad recibe tuplas de (padding_left, padding_right, padding_top, padding_bottom)
            pad_right = max_w - img.shape[2]
            pad_bottom = max_h - img.shape[1]
            
            padded_img = F.pad(img, (0, pad_right, 0, pad_bottom), value=0.0)
            padded_images.append(padded_img)
            
        # Ahora sí podemos usar stack porque todas miden lo mismo
        images_batched = torch.stack(padded_images, dim=0)
        
        # Los sketches suelen tener un tamaño fijo (ej. 224x224), así que stack normal
        sketches_batched = torch.stack(sketches, dim=0)
        
        return images_batched, sketches_batched, targets


def parse_dataset(json_path, config):
    """
    Parse JSON dataset and create train/val/test splits
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Organize classes
    class_to_queries = {}
    for cls in data['classes']:
        class_to_queries[cls['class_id']] = {
            'name': cls['class_name'],
            'queries': cls['queries']
        }
    
    # Organize samples
    samples = []
    for sample in data['samples']:
        # Filter boxes by score
        valid_boxes = [
            b for b in sample['boxes'] 
            if b['score'] >= config.BOX_SCORE_THRESHOLD
        ]
        
        if len(valid_boxes) > 0:
            samples.append({
                'sample_id': sample['sample_id'],
                'image_path': sample['page_path'],
                'boxes': valid_boxes,
                'class_id': sample['class_id'],
                'queries': class_to_queries[sample['class_id']]['queries']
            })
    
    print(f"Total samples after filtering: {len(samples)}")
    
    # Group samples by class for stratified split
    class_samples = {}
    for sample in samples:
        class_id = sample['class_id']
        if class_id not in class_samples:
            class_samples[class_id] = []
        class_samples[class_id].append(sample)
    
    # Stratified split
    train_samples = []
    val_samples = []
    test_samples = []
    
    for class_id, cls_samples in class_samples.items():
        random.shuffle(cls_samples)
        n = len(cls_samples)
        
        n_train = int(n * config.TRAIN_SPLIT)
        n_val = int(n * config.VAL_SPLIT)
        
        train_samples.extend(cls_samples[:n_train])
        val_samples.extend(cls_samples[n_train:n_train+n_val])
        test_samples.extend(cls_samples[n_train+n_val:])
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples, class_to_queries


def create_dataloaders(config):
    """
    Create train/val/test dataloaders
    """
    # Set random seed for reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    # Parse dataset
    train_samples, val_samples, test_samples, class_to_queries = parse_dataset(
        config.DATASET_JSON, config
    )
    
    # Create datasets
    train_dataset = QueryDetectionDataset(
        train_samples, class_to_queries, config, mode='train'
    )
    val_dataset = QueryDetectionDataset(
        val_samples, class_to_queries, config, mode='val'
    )
    test_dataset = QueryDetectionDataset(
        test_samples, class_to_queries, config, mode='test'
    )
    
    # Class weights for balancing (if enabled)
    if config.USE_CLASS_BALANCING:
        class_counts = {}
        for sample in train_samples:
            class_id = sample['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Inverse frequency weighting
        total = sum(class_counts.values())
        class_weights = {
            k: total / (len(class_counts) * v) 
            for k, v in class_counts.items()
        }
        
        # Sample weights
        sample_weights = [class_weights[s['class_id']] for s in train_samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights)
        )
    else:
        sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=QueryDetectionDataset.collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=QueryDetectionDataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=QueryDetectionDataset.collate_fn
    )
    
    return train_loader, val_loader, test_loader, class_to_queries