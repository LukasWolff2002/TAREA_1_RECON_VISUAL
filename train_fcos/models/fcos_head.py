"""
models/fcos_head.py

Cabeza de detección FCOS con FiLM conditioning por query sketch.

Estructura por nivel FPN:
  feature_map [B, 256, H, W]
      ↓ FiLM(query_emb)
  cls_branch (4 × conv-GN-ReLU) → sigmoid → [B, num_classes, H, W]
  reg_branch (4 × conv-GN-ReLU) → exp()   → [B, 4, H, W]  (ltrb)
  ctr_branch (desde reg features)→ sigmoid → [B, 1, H, W]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Genera (γ, β) desde un embedding de query y los aplica al feature map.
        out = γ * x + β
    """

    def __init__(self, query_dim: int, feat_channels: int):
        super().__init__()
        # Proyección lineal: query → [γ, β] por canal
        self.proj = nn.Linear(query_dim, 2 * feat_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # Inicializar γ=1, β=0 → sin efecto al inicio del entrenamiento
        with torch.no_grad():
            self.proj.bias[:feat_channels] = 1.0   # γ init = 1

    def forward(self, x: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """
        x:         [B, C, H, W]
        query_emb: [B, query_dim]
        Returns:   [B, C, H, W]
        """
        params = self.proj(query_emb)          # [B, 2C]
        gamma  = params[:, :x.shape[1]]        # [B, C]
        beta   = params[:, x.shape[1]:]        # [B, C]
        # Reshape para broadcast: [B, C, 1, 1]
        gamma  = gamma.unsqueeze(-1).unsqueeze(-1)
        beta   = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv 3x3 + GroupNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.ReLU(inplace=True),
    )


class FCOSHead(nn.Module):
    """
    Cabeza FCOS compartida entre todos los niveles FPN.
    Los pesos de los conv son shared; solo la escala (scale) es por nivel.
    """

    def __init__(
        self,
        in_channels:  int = 256,
        num_convs:    int = 4,
        num_classes:  int = 22,
        query_dim:    int = 1024,
        strides:      list = None,
        norm_on_bbox: bool = True,
        centerness_on_reg: bool = True,
    ):
        super().__init__()
        self.num_classes       = num_classes
        self.strides           = strides or [4, 8, 16, 32, 64]
        self.norm_on_bbox      = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg

        # ── FiLM conditioning (uno por nivel para más expresividad) ──────────
        self.film_cls = nn.ModuleList([
            FiLMLayer(query_dim, in_channels)
            for _ in self.strides
        ])
        self.film_reg = nn.ModuleList([
            FiLMLayer(query_dim, in_channels)
            for _ in self.strides
        ])

        # ── Clasificación branch (shared weights) ────────────────────────────
        cls_convs = []
        for i in range(num_convs):
            cls_convs.append(conv_block(in_channels, in_channels))
        self.cls_convs = nn.Sequential(*cls_convs)

        # ── Regresión branch (shared weights) ────────────────────────────────
        reg_convs = []
        for i in range(num_convs):
            reg_convs.append(conv_block(in_channels, in_channels))
        self.reg_convs = nn.Sequential(*reg_convs)

        # ── Capas de salida ───────────────────────────────────────────────────
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4,           kernel_size=3, padding=1)
        self.ctr_pred = nn.Conv2d(in_channels, 1,           kernel_size=3, padding=1)

        # Escala aprendible por nivel (estabiliza el entrenamiento)
        self.scales = nn.ModuleList([
            nn.Parameter(torch.ones(1)) if False else Scale(init_val=1.0)
            for _ in self.strides
        ])

        self._init_weights()

    def _init_weights(self):
        # Inicialización estándar FCOS
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Bias del predictor de cls: inicializar con prior de probabilidad
        # para que sigmoid(bias) ≈ 0.01 → evita gradientes explosivos al inicio
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward_single_level(
        self,
        feat:      torch.Tensor,
        query_emb: torch.Tensor,
        level_idx: int,
    ):
        """
        Procesa UN nivel FPN.
        feat:      [B, C, H, W]
        query_emb: [B, query_dim]
        Returns: cls_logits [B, num_cls, H, W]
                 bbox_pred  [B, 4, H, W]   (ltrb positivos)
                 ctr_logits [B, 1, H, W]
        """
        # FiLM conditioning sobre clasificación
        cls_feat = self.film_cls[level_idx](feat, query_emb)
        cls_feat = self.cls_convs(cls_feat)
        cls_logits = self.cls_pred(cls_feat)

        # FiLM conditioning sobre regresión
        reg_feat = self.film_reg[level_idx](feat, query_emb)
        reg_feat = self.reg_convs(reg_feat)
        bbox_pred = self.scales[level_idx](self.reg_pred(reg_feat))
        bbox_pred = F.relu(bbox_pred)  # distancias siempre ≥ 0

        # Centerness desde las features de regresión
        ctr_logits = self.ctr_pred(reg_feat)

        return cls_logits, bbox_pred, ctr_logits

    def forward(self, features: list, query_emb: torch.Tensor):
        """
        features:  [P2, P3, P4, P5, P6]  (lista de tensores)
        query_emb: [B, query_dim]
        Returns: listas de longitud num_levels:
            all_cls   → cada elem [B, num_cls, H_i, W_i]
            all_bbox  → cada elem [B, 4, H_i, W_i]
            all_ctr   → cada elem [B, 1, H_i, W_i]
        """
        all_cls, all_bbox, all_ctr = [], [], []
        for i, feat in enumerate(features):
            cls_l, bbox_l, ctr_l = self.forward_single_level(feat, query_emb, i)
            all_cls.append(cls_l)
            all_bbox.append(bbox_l)
            all_ctr.append(ctr_l)
        return all_cls, all_bbox, all_ctr

    def decode_predictions(
        self,
        all_cls:  list,
        all_bbox: list,
        all_ctr:  list,
        img_shape: tuple,
        score_threshold: float = 0.05,
    ) -> list:
        """
        Convierte las salidas raw a detecciones en coordenadas de imagen.
        Retorna lista de dicts {boxes, scores, labels} por imagen del batch.
        """
        B = all_cls[0].shape[0]
        H_img, W_img = img_shape

        # Acumular predicciones de todos los niveles
        batch_boxes  = [[] for _ in range(B)]
        batch_scores = [[] for _ in range(B)]
        batch_labels = [[] for _ in range(B)]

        for lvl_idx, (cls_l, bbox_l, ctr_l) in enumerate(zip(all_cls, all_bbox, all_ctr)):
            stride = self.strides[lvl_idx]
            B, C, H, W = cls_l.shape

            # Generar puntos (centros de cada celda) en coordenadas de imagen
            ys = (torch.arange(H, device=cls_l.device).float() + 0.5) * stride
            xs = (torch.arange(W, device=cls_l.device).float() + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [H*W, 2]

            # Scores combinados: cls * centerness
            cls_scores = cls_l.sigmoid()   # [B, C, H, W]
            ctr_scores = ctr_l.sigmoid()   # [B, 1, H, W]
            scores     = (cls_scores * ctr_scores).permute(0, 2, 3, 1)  # [B, H, W, C]
            scores     = scores.reshape(B, H * W, C)

            # Regresión: ltrb
            if self.norm_on_bbox:
                bbox = bbox_l * stride
            else:
                bbox = bbox_l
            bbox = bbox.permute(0, 2, 3, 1).reshape(B, H * W, 4)  # [B, H*W, 4]

            for b in range(B):
                # Filtrar por score threshold
                max_scores, max_labels = scores[b].max(dim=-1)  # [H*W]
                keep_mask = max_scores > score_threshold

                if keep_mask.sum() == 0:
                    continue

                kp = keep_mask.nonzero(as_tuple=False).squeeze(1)
                pts_k   = points[kp]            # [K, 2]
                bbox_k  = bbox[b][kp]           # [K, 4]
                scores_k = max_scores[kp]       # [K]
                labels_k = max_labels[kp]       # [K]

                # Convertir ltrb a xyxy
                x1 = (pts_k[:, 0] - bbox_k[:, 0]).clamp(0, W_img)
                y1 = (pts_k[:, 1] - bbox_k[:, 1]).clamp(0, H_img)
                x2 = (pts_k[:, 0] + bbox_k[:, 2]).clamp(0, W_img)
                y2 = (pts_k[:, 1] + bbox_k[:, 3]).clamp(0, H_img)
                boxes_k = torch.stack([x1, y1, x2, y2], dim=-1)

                batch_boxes[b].append(boxes_k)
                batch_scores[b].append(scores_k)
                batch_labels[b].append(labels_k)

        # Concatenar todos los niveles
        results = []
        for b in range(B):
            if batch_boxes[b]:
                boxes  = torch.cat(batch_boxes[b],  dim=0)
                scores = torch.cat(batch_scores[b], dim=0)
                labels = torch.cat(batch_labels[b], dim=0)
            else:
                device = all_cls[0].device
                boxes  = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,),   device=device)
                labels = torch.zeros((0,),   dtype=torch.long, device=device)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})

        return results


class Scale(nn.Module):
    """Escala aprendible por nivel (inicializada a 1)."""
    def __init__(self, init_val: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_val], dtype=torch.float))

    def forward(self, x):
        return x * self.scale
