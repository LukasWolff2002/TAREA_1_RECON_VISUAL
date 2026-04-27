"""
utils/box_utils.py
Operaciones geométricas sobre bounding boxes.
Formato esperado: [x1, y1, x2, y2] (xyxy).
"""

import torch
import torch.nn.functional as F


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Área de cada box. boxes: [N, 4] en formato xyxy."""
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * \
           (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU entre dos conjuntos de boxes.
    boxes1: [N, 4], boxes2: [M, 4]
    Returns: [N, M]
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * \
            (inter_y2 - inter_y1).clamp(min=0)  # [N, M]

    union = area1[:, None] + area2[None, :] - inter  # [N, M]
    return inter / union.clamp(min=1e-6)


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU entre pares de boxes (N == M).
    boxes1: [N, 4], boxes2: [N, 4]
    Returns: [N]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * \
            (inter_y2 - inter_y1).clamp(min=0)

    union = area1 + area2 - inter
    iou   = inter / union.clamp(min=1e-6)

    # Caja envolvente más pequeña que contiene ambos boxes
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * \
               (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


def compute_centerness_targets(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Calcula el target de centerness desde las distancias l, t, r, b.
    ltrb: [N, 4] con (left, top, right, bottom)
    Returns: [N]
    """
    l, t, r, b = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
    centerness = torch.sqrt(
        (torch.min(l, r) / torch.max(l, r).clamp(min=1e-6)) *
        (torch.min(t, b) / torch.max(t, b).clamp(min=1e-6))
    )
    return centerness.clamp(min=0, max=1)


def ltrb_to_xyxy(ltrb: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Convierte predicciones FCOS (l, t, r, b) relativas a puntos a formato xyxy.
    ltrb:   [N, 4]
    points: [N, 2] con (x, y) del centro de cada celda
    Returns: [N, 4] en xyxy
    """
    x1 = points[:, 0] - ltrb[:, 0]
    y1 = points[:, 1] - ltrb[:, 1]
    x2 = points[:, 0] + ltrb[:, 2]
    y2 = points[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_ltrb(boxes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Convierte boxes xyxy a distancias (l, t, r, b) desde los puntos.
    boxes:  [N, 4]
    points: [N, 2]
    Returns: [N, 4]
    """
    l = points[:, 0] - boxes[:, 0]
    t = points[:, 1] - boxes[:, 1]
    r = boxes[:, 2] - points[:, 0]
    b = boxes[:, 3] - points[:, 1]
    return torch.stack([l, t, r, b], dim=-1)


def batch_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
              iou_threshold: float = 0.5) -> torch.Tensor:
    """
    NMS por clase (class-aware NMS).
    boxes:  [N, 4], scores: [N], labels: [N]
    Returns: índices kept [K]
    """
    from torchvision.ops import nms
    keep_all = []
    unique_labels = labels.unique()
    for cls in unique_labels:
        mask = labels == cls
        idx  = mask.nonzero(as_tuple=False).squeeze(1)
        kept = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.append(idx[kept])
    if not keep_all:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    return torch.cat(keep_all)


def clip_boxes_to_image(boxes: torch.Tensor, size: tuple) -> torch.Tensor:
    """Clipa boxes al tamaño de imagen (H, W)."""
    H, W = size
    boxes[:, 0].clamp_(min=0, max=W)
    boxes[:, 1].clamp_(min=0, max=H)
    boxes[:, 2].clamp_(min=0, max=W)
    boxes[:, 3].clamp_(min=0, max=H)
    return boxes
