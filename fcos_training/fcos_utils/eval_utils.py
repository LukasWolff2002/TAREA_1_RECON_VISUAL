"""
Utilidades para entrenamiento y evaluación
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        iou_threshold: float
    Returns:
        keep: indices de boxes a mantener
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i.item())
        
        if len(order) == 1:
            break
        
        # IoU con el resto
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        # Mantener solo los que tienen IoU < threshold
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def decode_predictions(predictions, config):
    """
    Decodificar predicciones de FCOS a bounding boxes
    Args:
        predictions: List de dicts con cls, reg, ctr por nivel
        config: Config object
    Returns:
        List de detecciones por imagen: [{boxes, scores, labels}]
    """
    batch_size = predictions[0]['cls'].shape[0]
    device = predictions[0]['cls'].device
    
    batch_detections = []
    
    for batch_idx in range(batch_size):
        all_boxes = []
        all_scores = []
        
        for level_idx, pred in enumerate(predictions):
            cls_pred = pred['cls'][batch_idx, 0]  # (H, W)
            reg_pred = pred['reg'][batch_idx]  # (4, H, W)
            ctr_pred = pred['ctr'][batch_idx, 0]  # (H, W)
            stride = pred['stride']
            
            h, w = cls_pred.shape
            
            # Scores = cls * centerness
            cls_scores = torch.sigmoid(cls_pred)
            ctr_scores = torch.sigmoid(ctr_pred)
            scores = torch.sqrt(cls_scores * ctr_scores)
            
            # Filtrar por score threshold
            mask = scores > config.SCORE_THRESHOLD
            
            if mask.sum() == 0:
                continue
            
            # Grid locations
            shifts_x = torch.arange(0, w, device=device) * stride + stride // 2
            shifts_y = torch.arange(0, h, device=device) * stride + stride // 2
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            locations = torch.stack([shift_x, shift_y], dim=-1)  # (H, W, 2)
            
            # Flatten
            locations_flat = locations.view(-1, 2)
            reg_flat = reg_pred.permute(1, 2, 0).reshape(-1, 4)  # (H*W, 4)
            scores_flat = scores.flatten()
            mask_flat = mask.flatten()
            
            # Filtered
            locations_pos = locations_flat[mask_flat]
            reg_pos = reg_flat[mask_flat]
            scores_pos = scores_flat[mask_flat]
            
            # Decode boxes (ltrb -> xyxy)
            boxes = torch.zeros(len(reg_pos), 4, device=device)
            boxes[:, 0] = locations_pos[:, 0] - reg_pos[:, 0]  # x1
            boxes[:, 1] = locations_pos[:, 1] - reg_pos[:, 1]  # y1
            boxes[:, 2] = locations_pos[:, 0] + reg_pos[:, 2]  # x2
            boxes[:, 3] = locations_pos[:, 1] + reg_pos[:, 3]  # y2
            
            all_boxes.append(boxes)
            all_scores.append(scores_pos)
        
        if len(all_boxes) > 0:
            # Concatenate all levels
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            
            # NMS
            keep = apply_nms(all_boxes, all_scores, config.NMS_THRESHOLD)
            
            if len(keep) > config.MAX_DETECTIONS_PER_IMAGE:
                keep = keep[:config.MAX_DETECTIONS_PER_IMAGE]
            
            final_boxes = all_boxes[keep]
            final_scores = all_scores[keep]
        else:
            final_boxes = torch.zeros(0, 4, device=device)
            final_scores = torch.zeros(0, device=device)
        
        batch_detections.append({
            'boxes': final_boxes,
            'scores': final_scores
        })
    
    return batch_detections


def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Compute Average Precision para una imagen
    """
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    
    if len(pred_boxes) == 0:
        return 0.0
    
    # Sort by score
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Compute IoU matrix
    def compute_iou_numpy(boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        iou = inter / (area1[:, None] + area2 - inter + 1e-6)
        return iou
    
    ious = compute_iou_numpy(pred_boxes, gt_boxes)
    
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
    
    return ap


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_model(model, dataloader, config, device):
    """
    Evaluate model on validation/test set
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, sketches, targets in dataloader:
            images = images.to(device)
            sketches = sketches.to(device)
            
            # Forward
            predictions = model(images, sketches)
            
            # Decode
            detections = decode_predictions(predictions, config)
            
            for det, target in zip(detections, targets):
                all_predictions.append({
                    'boxes': det['boxes'].cpu().numpy(),
                    'scores': det['scores'].cpu().numpy()
                })
                all_targets.append({
                    'boxes': target['boxes'].cpu().numpy()
                })
    
    # Compute mAP
    aps = []
    for pred, target in zip(all_predictions, all_targets):
        ap = compute_ap(
            pred['boxes'], pred['scores'], target['boxes'],
            iou_threshold=0.5
        )
        aps.append(ap)
    
    mAP = np.mean(aps) if len(aps) > 0 else 0.0
    
    return {
        'mAP': mAP,
        'num_images': len(all_predictions)
    }