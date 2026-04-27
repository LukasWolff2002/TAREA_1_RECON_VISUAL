"""
Loss functions para FCOS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou(boxes1, boxes2):
    """
    Compute IoU entre dos sets de boxes
    Args:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]
    Returns:
        iou: (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    iou = inter / (area1[:, None] + area2 - inter + 1e-6)
    return iou


def compute_giou(boxes1, boxes2):
    """
    Generalized IoU
    """
    iou = compute_iou(boxes1, boxes2)
    
    # Enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing_area = wh[:, :, 0] * wh[:, :, 1]
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - iou * (area1[:, None] + area2)
    
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N,) logits
            targets: (N,) binary labels {0, 1}
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        return loss.mean()


class IoULoss(nn.Module):
    """
    IoU-based loss for bbox regression
    """
    def __init__(self, loss_type='giou'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (N, 4) [x1, y1, x2, y2]
            target_boxes: (N, 4) [x1, y1, x2, y2]
        """
        if self.loss_type == 'iou':
            iou = compute_iou(pred_boxes, target_boxes).diagonal()
            loss = 1 - iou
        elif self.loss_type == 'giou':
            giou = compute_giou(pred_boxes, target_boxes).diagonal()
            loss = 1 - giou
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss.mean()


class FCOSLoss(nn.Module):
    """
    Complete FCOS Loss
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.focal_loss = FocalLoss(
            alpha=config.FOCAL_LOSS_ALPHA,
            gamma=config.FOCAL_LOSS_GAMMA
        )
        self.iou_loss = IoULoss(loss_type=config.IOU_LOSS_TYPE)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def prepare_targets(self, predictions, targets):
        """
        Assign GT boxes to FPN levels y generar targets por píxel
        """
        all_level_targets = []
        
        for level_idx, pred in enumerate(predictions):
            stride = pred['stride']
            device = pred['cls'].device
            
            batch_targets = []
            for batch_idx, target in enumerate(targets):
                boxes = target['boxes']  # (N, 4) xyxy
                
                if len(boxes) == 0:
                    # No objects
                    h, w = pred['cls'].shape[-2:]
                    batch_targets.append({
                        'labels': torch.zeros(h, w, device=device),
                        'reg_targets': torch.zeros(h, w, 4, device=device),
                        'centerness_targets': torch.zeros(h, w, device=device),
                        'mask': torch.zeros(h, w, dtype=torch.bool, device=device)
                    })
                    continue
                
                # Compute box areas
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                # Assign to this level based on size
                size_range = self.config.FPN_SIZE_RANGES[level_idx]
                valid_mask = (areas >= size_range[0]**2) & (areas < size_range[1]**2)
                
                level_boxes = boxes[valid_mask]
                
                # Generate per-pixel targets
                h, w = pred['cls'].shape[-2:]
                labels = torch.zeros(h, w, device=device)
                reg_targets = torch.zeros(h, w, 4, device=device)
                centerness_targets = torch.zeros(h, w, device=device)
                mask = torch.zeros(h, w, dtype=torch.bool, device=device)
                
                if len(level_boxes) > 0:
                    # Grid coordinates
                    shifts_x = torch.arange(0, w, device=device) * stride + stride // 2
                    shifts_y = torch.arange(0, h, device=device) * stride + stride // 2
                    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
                    locations = torch.stack([shift_x, shift_y], dim=-1)  # (H, W, 2)
                    
                    # Para cada box, encontrar píxeles dentro
                    # En loss.py, reemplaza líneas 171-179:

                    for box in level_boxes:
                        x1, y1, x2, y2 = box
                        
                        # Calcular tamaño de la box
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_size = torch.sqrt(box_w * box_h)
                        
                        # Radius proporcional al tamaño de la box
                        # Para boxes pequeñas, usar más del 80% de la box
                        # Para boxes grandes, usar menos (center sampling más estricto)
                        if box_size < 64:
                            radius_ratio = 0.8  # 80% de la box
                        elif box_size < 128:
                            radius_ratio = 0.6  # 60% de la box
                        else:
                            radius_ratio = 0.4  # 40% de la box (más estricto)
                        
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        radius = box_size * radius_ratio  # ← Basado en el tamaño real
                        
                        # ... resto del código igual ...
                        
                        # Píxeles dentro del radius del centro
                        dist_to_center = torch.sqrt(
                            (locations[..., 0] - cx)**2 + (locations[..., 1] - cy)**2
                        )
                        in_center = dist_to_center < radius
                        
                        # Píxeles dentro de la box
                        in_box = (
                            (locations[..., 0] >= x1) & (locations[..., 0] <= x2) &
                            (locations[..., 1] >= y1) & (locations[..., 1] <= y2)
                        )
                        
                        valid = in_box & in_center
                        
                        # Regression targets (l, t, r, b)
                        l = locations[..., 0] - x1
                        t = locations[..., 1] - y1
                        r = x2 - locations[..., 0]
                        b = y2 - locations[..., 1]
                        
                        reg = torch.stack([l, t, r, b], dim=-1)  # (H, W, 4)
                        
                        # Centerness
                        lr_min = torch.min(l, r)
                        lr_max = torch.max(l, r)
                        tb_min = torch.min(t, b)
                        tb_max = torch.max(t, b)
                        centerness = torch.sqrt(
                            (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
                        )
                        
                        # Update targets
                        labels[valid] = 1
                        reg_targets[valid] = reg[valid]
                        centerness_targets[valid] = centerness[valid]
                        mask[valid] = True
                
                batch_targets.append({
                    'labels': labels,
                    'reg_targets': reg_targets,
                    'centerness_targets': centerness_targets,
                    'mask': mask
                })
            
            all_level_targets.append(batch_targets)
        
        return all_level_targets
    
    def forward(self, predictions, targets):
        """
        Compute total loss
        """
        # Prepare targets for each FPN level
        level_targets = self.prepare_targets(predictions, targets)
        
        total_cls_loss = 0
        total_reg_loss = 0
        total_ctr_loss = 0
        num_pos = 0
        
        for level_idx, (pred, batch_targets) in enumerate(zip(predictions, level_targets)):
            cls_pred = pred['cls']  # (B, 1, H, W)
            reg_pred = pred['reg']  # (B, 4, H, W)
            ctr_pred = pred['ctr']  # (B, 1, H, W)
            stride = pred['stride']
            
            batch_size = cls_pred.shape[0]
            
            for batch_idx in range(batch_size):
                target = batch_targets[batch_idx]
                
                labels = target['labels']  # (H, W)
                reg_targets = target['reg_targets']  # (H, W, 4)
                ctr_targets = target['centerness_targets']  # (H, W)
                mask = target['mask']  # (H, W)
                
                # Flatten
                cls_flat = cls_pred[batch_idx, 0].flatten()  # (H*W,)
                reg_flat = reg_pred[batch_idx].permute(1, 2, 0).flatten(0, 1)  # (H*W, 4)
                ctr_flat = ctr_pred[batch_idx, 0].flatten()  # (H*W,)
                
                labels_flat = labels.flatten()
                reg_targets_flat = reg_targets.flatten(0, 1)  # (H*W, 4)
                ctr_targets_flat = ctr_targets.flatten()
                mask_flat = mask.flatten()
                
                # Classification loss (all pixels)
                cls_loss = self.focal_loss(cls_flat, labels_flat)
                total_cls_loss += cls_loss
                
                # Regression y centerness loss (solo píxeles positivos)
                if mask_flat.sum() > 0:
                    pos_reg_pred = reg_flat[mask_flat]
                    pos_reg_target = reg_targets_flat[mask_flat]
                    pos_ctr_pred = ctr_flat[mask_flat]
                    pos_ctr_target = ctr_targets_flat[mask_flat]
                    
                    # Convertir ltrb a xyxy para IoU loss
                    h, w = labels.shape
                    shifts_x = torch.arange(0, w, device=labels.device) * stride + stride // 2
                    shifts_y = torch.arange(0, h, device=labels.device) * stride + stride // 2
                    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
                    locations = torch.stack([shift_x, shift_y], dim=-1).flatten(0, 1)  # (H*W, 2)
                    
                    pos_locations = locations[mask_flat]
                    
                    # Pred boxes
                    pred_boxes = torch.zeros(len(pos_reg_pred), 4, device=labels.device)
                    pred_boxes[:, 0] = pos_locations[:, 0] - pos_reg_pred[:, 0]  # x1
                    pred_boxes[:, 1] = pos_locations[:, 1] - pos_reg_pred[:, 1]  # y1
                    pred_boxes[:, 2] = pos_locations[:, 0] + pos_reg_pred[:, 2]  # x2
                    pred_boxes[:, 3] = pos_locations[:, 1] + pos_reg_pred[:, 3]  # y2
                    
                    # Target boxes
                    target_boxes = torch.zeros(len(pos_reg_target), 4, device=labels.device)
                    target_boxes[:, 0] = pos_locations[:, 0] - pos_reg_target[:, 0]
                    target_boxes[:, 1] = pos_locations[:, 1] - pos_reg_target[:, 1]
                    target_boxes[:, 2] = pos_locations[:, 0] + pos_reg_target[:, 2]
                    target_boxes[:, 3] = pos_locations[:, 1] + pos_reg_target[:, 3]
                    
                    # IoU Loss
                    reg_loss = self.iou_loss(pred_boxes, target_boxes)
                    total_reg_loss += reg_loss
                    
                    # Centerness Loss
                    ctr_loss = self.bce_loss(pos_ctr_pred, pos_ctr_target)
                    total_ctr_loss += ctr_loss
                    
                    num_pos += mask_flat.sum().item()
        
        # Average over batch y levels
        num_levels = len(predictions)
        batch_size = predictions[0]['cls'].shape[0]
        normalizer = batch_size * num_levels
        
        total_cls_loss = total_cls_loss / normalizer
        
        if num_pos > 0:
            total_reg_loss = total_reg_loss / num_pos
            total_ctr_loss = total_ctr_loss / num_pos
        
        # Weighted sum
        total_loss = (
            self.config.LOSS_WEIGHT_CLS * total_cls_loss +
            self.config.LOSS_WEIGHT_REG * total_reg_loss +
            self.config.LOSS_WEIGHT_CTR * total_ctr_loss
        )
        
        return {
            'total': total_loss,
            'cls': total_cls_loss,
            'reg': total_reg_loss,
            'ctr': total_ctr_loss,
            'num_pos': num_pos
        }