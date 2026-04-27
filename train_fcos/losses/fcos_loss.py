"""
losses/fcos_loss.py

Función de pérdida FCOS completa:
  L = λ_cls * L_focal  +  λ_bbox * L_giou  +  λ_ctr * L_centerness

Asignación de targets:
  1. Un punto (x, y) es positivo para GT box si está dentro de la box.
  2. La distancia max(l,t,r,b) debe caer dentro del rango de regresión del nivel.
  3. Si un punto cae en múltiples boxes, se asigna a la de menor área.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_fcos.utils.box_utils import compute_centerness_targets, box_giou


def sigmoid_focal_loss(
    inputs:  torch.Tensor,   # logits [N, C]
    targets: torch.Tensor,   # one-hot [N, C]
    alpha:   float = 0.25,
    gamma:   float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Focal Loss estándar."""
    p      = inputs.sigmoid()
    p_t    = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss   = - alpha_t * (1 - p_t) ** gamma * \
               torch.log(p_t.clamp(min=1e-7))

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    return loss


class FCOSLoss(nn.Module):
    """
    Calcula la pérdida FCOS dados los outputs del head y los targets GT.
    """

    def __init__(
        self,
        num_classes:      int   = 22,
        strides:          list  = None,
        regress_ranges:   tuple = None,
        focal_alpha:      float = 0.25,
        focal_gamma:      float = 2.0,
        lambda_cls:       float = 1.0,
        lambda_bbox:      float = 1.0,
        lambda_ctr:       float = 1.0,
        norm_on_bbox:     bool  = True,
        class_weights:    torch.Tensor = None,
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.strides        = strides      or [4, 8, 16, 32, 64]
        self.regress_ranges = regress_ranges or (
            (0, 32), (32, 64), (64, 128), (128, 256), (256, 1e8)
        )
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma
        self.lambda_cls   = lambda_cls
        self.lambda_bbox  = lambda_bbox
        self.lambda_ctr   = lambda_ctr
        self.norm_on_bbox = norm_on_bbox

        # Pesos por clase (inverso de frecuencia)
        self.register_buffer("class_weights",
                             class_weights if class_weights is not None
                             else torch.ones(num_classes))

    # ─────────────────────────────────────────────────────────────────────────
    def _get_points(self, all_cls: list, device: torch.device) -> list:
        """Genera los puntos centrales de cada nivel FPN."""
        all_points = []
        for lvl, stride in enumerate(self.strides):
            _, _, H, W = all_cls[lvl].shape
            ys = (torch.arange(H, device=device).float() + 0.5) * stride
            xs = (torch.arange(W, device=device).float() + 0.5) * stride
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pts = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [H*W, 2]
            all_points.append(pts)
        return all_points   # lista de [H_i*W_i, 2]

    # ─────────────────────────────────────────────────────────────────────────
    def _assign_targets(
        self,
        all_points: list,
        gt_boxes:   torch.Tensor,   # [M, 4] xyxy
        gt_labels:  torch.Tensor,   # [M]
    ):
        """
        Asigna gt a cada punto de todos los niveles.
        Returns:
            cls_targets:  [N_total] con -1=ignorar, 0..C-1 clases, C=fondo
            bbox_targets: [N_total, 4] en coordenadas absolutas (ltrb)
            level_mask:   [N_total] nivel al que pertenece cada punto
        """
        points_all = torch.cat(all_points, dim=0)   # [N_total, 2]
        N = points_all.shape[0]
        M = gt_boxes.shape[0]

        if M == 0:
            return (
                torch.full((N,), self.num_classes, dtype=torch.long,
                           device=points_all.device),
                torch.zeros((N, 4), device=points_all.device),
            )

        # Rango de cada punto según nivel
        lvl_ranges = []
        for lvl, pts in enumerate(all_points):
            r_min, r_max = self.regress_ranges[lvl]
            lvl_ranges.append(torch.full((len(pts),), lvl,
                                         device=points_all.device))
        lvl_idx = torch.cat(lvl_ranges)  # [N_total]

        # Distancias l, t, r, b desde cada punto a cada GT box
        x = points_all[:, 0][:, None]   # [N, 1]
        y = points_all[:, 1][:, None]   # [N, 1]
        x1, y1, x2, y2 = (gt_boxes[:, i][None, :] for i in range(4))  # [1, M]

        l = x - x1   # [N, M]
        t = y - y1
        r = x2 - x
        b = y2 - y

        ltrb = torch.stack([l, t, r, b], dim=-1)   # [N, M, 4]

        # Punto dentro de la box → todas las distancias > 0
        inside_box = ltrb.min(dim=-1).values > 0   # [N, M]

        # Restricción de rango: max(l,t,r,b) dentro del rango del nivel
        max_ltrb = ltrb.max(dim=-1).values          # [N, M]
        for lvl, (r_min, r_max) in enumerate(self.regress_ranges):
            lvl_mask = (lvl_idx == lvl)[:, None].expand_as(max_ltrb)
            in_range = (max_ltrb >= r_min) & (max_ltrb <= r_max)
            inside_box = inside_box & (in_range | ~lvl_mask)
            # Equivalente: si el punto es de nivel lvl, debe estar en su rango

        # Resolver ambigüedad: elegir GT de menor área si hay múltiples
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                   (gt_boxes[:, 3] - gt_boxes[:, 1])   # [M]
        areas = gt_areas[None, :].expand(N, M)
        areas = areas.clone()
        areas[~inside_box] = float("inf")

        min_areas, min_idx = areas.min(dim=-1)   # [N]
        assigned = min_areas < float("inf")      # [N] — puntos positivos

        cls_targets  = torch.full((N,), self.num_classes, dtype=torch.long,
                                   device=points_all.device)
        bbox_targets = torch.zeros((N, 4), device=points_all.device)

        if assigned.sum() > 0:
            cls_targets[assigned]  = gt_labels[min_idx[assigned]]
            assigned_ltrb = ltrb[assigned, min_idx[assigned]]    # [K, 4]
            bbox_targets[assigned] = assigned_ltrb

        return cls_targets, bbox_targets

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        all_cls:   list,    # [P2..P6] cada [B, C, H_i, W_i]
        all_bbox:  list,    # [P2..P6] cada [B, 4, H_i, W_i]
        all_ctr:   list,    # [P2..P6] cada [B, 1, H_i, W_i]
        targets:   list,    # lista de dicts por imagen: {boxes, labels}
    ) -> dict:

        device = all_cls[0].device
        B      = all_cls[0].shape[0]

        # Puntos de todos los niveles (compartidos para todo el batch)
        all_points = self._get_points(all_cls, device)  # lista de [H_i*W_i, 2]

        # Flatten predicciones: [B, N_total, *]
        flat_cls  = torch.cat(
            [c.permute(0,2,3,1).reshape(B, -1, self.num_classes) for c in all_cls],
            dim=1
        )  # [B, N_total, C]
        flat_bbox = torch.cat(
            [b.permute(0,2,3,1).reshape(B, -1, 4) for b in all_bbox],
            dim=1
        )  # [B, N_total, 4]
        flat_ctr  = torch.cat(
            [c.permute(0,2,3,1).reshape(B, -1, 1) for c in all_ctr],
            dim=1
        )  # [B, N_total, 1]

        # Escalar bbox si norm_on_bbox
        if self.norm_on_bbox:
            strides_tensor = []
            for lvl, pts in enumerate(all_points):
                strides_tensor.append(
                    torch.full((len(pts),), self.strides[lvl], device=device)
                )
            strides_per_point = torch.cat(strides_tensor)[None, :, None]  # [1, N, 1]
            flat_bbox = flat_bbox * strides_per_point

        # Acumular pérdidas
        loss_cls_total  = torch.tensor(0.0, device=device)
        loss_bbox_total = torch.tensor(0.0, device=device)
        loss_ctr_total  = torch.tensor(0.0, device=device)
        n_pos_total     = 0

        for b in range(B):
            gt_boxes  = targets[b]["boxes"].to(device)   # [M, 4]
            gt_labels = targets[b]["labels"].to(device)  # [M]

            cls_t, bbox_t = self._assign_targets(all_points, gt_boxes, gt_labels)
            # cls_t:  [N_total]  — num_classes = background
            # bbox_t: [N_total, 4]

            pos_mask = cls_t < self.num_classes   # [N_total]
            n_pos    = pos_mask.sum().item()
            n_pos_total += max(n_pos, 1)

            # ── Focal Loss (clasificación) ─────────────────────────────────
            cls_logits = flat_cls[b]   # [N_total, C]
            cls_oh     = F.one_hot(
                cls_t.clamp(0, self.num_classes - 1),
                num_classes=self.num_classes
            ).float()
            cls_oh[~pos_mask] = 0.0   # puntos de fondo → all-zero target

            loss_cls = sigmoid_focal_loss(
                cls_logits, cls_oh,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction="sum"
            )
            loss_cls_total += loss_cls

            if n_pos == 0:
                continue

            # ── GIoU Loss (regresión) ─────────────────────────────────────
            pred_ltrb = flat_bbox[b][pos_mask]     # [K, 4]
            tgt_ltrb  = bbox_t[pos_mask]           # [K, 4]

            # Necesitamos convertir ltrb a xyxy para GIoU
            points_pos = torch.cat(all_points, dim=0)[pos_mask]  # [K, 2]
            pred_x1 = (points_pos[:, 0] - pred_ltrb[:, 0]).clamp(min=0)
            pred_y1 = (points_pos[:, 1] - pred_ltrb[:, 1]).clamp(min=0)
            pred_x2 = (points_pos[:, 0] + pred_ltrb[:, 2])
            pred_y2 = (points_pos[:, 1] + pred_ltrb[:, 3])
            pred_boxes_xy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

            tgt_x1 = (points_pos[:, 0] - tgt_ltrb[:, 0])
            tgt_y1 = (points_pos[:, 1] - tgt_ltrb[:, 1])
            tgt_x2 = (points_pos[:, 0] + tgt_ltrb[:, 2])
            tgt_y2 = (points_pos[:, 1] + tgt_ltrb[:, 3])
            tgt_boxes_xy = torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=-1)

            giou  = box_giou(pred_boxes_xy, tgt_boxes_xy)   # [K]
            loss_bbox_total += (1 - giou).sum()

            # ── Centerness Loss ───────────────────────────────────────────
            pred_ctr = flat_ctr[b][pos_mask].squeeze(-1)    # [K]
            tgt_ctr  = compute_centerness_targets(tgt_ltrb) # [K]
            loss_ctr = F.binary_cross_entropy_with_logits(
                pred_ctr, tgt_ctr, reduction="sum"
            )
            loss_ctr_total += loss_ctr

        # Normalizar por número total de positivos
        denom = max(n_pos_total, 1)
        loss_cls  = self.lambda_cls  * loss_cls_total  / denom
        loss_bbox = self.lambda_bbox * loss_bbox_total / denom
        loss_ctr  = self.lambda_ctr  * loss_ctr_total  / denom
        total     = loss_cls + loss_bbox + loss_ctr

        return {
            "loss":      total,
            "loss_cls":  loss_cls,
            "loss_bbox": loss_bbox,
            "loss_ctr":  loss_ctr,
            "n_pos":     n_pos_total / B,
        }
