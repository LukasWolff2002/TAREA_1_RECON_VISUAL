"""
models/detector.py

Modelo de detección completo:
    query_img → iDocQueryEncoder → query_emb [B, 1024]
    page_img  → iDocBackbone     → [C2, C3, C4, C5]
                   → FPN         → [P2, P3, P4, P5, P6]
                   → FCOSHead(query_emb) → cls, bbox, centerness
"""

import torch
import torch.nn as nn

# Importaciones absolutas (asumiendo que TRAIN_FCOS_DIR está en el sys.path)
from train_fcos.models.backbone   import iDocBackbone, iDocQueryEncoder
from train_fcos.models.fpn        import FPN
from train_fcos.models.fcos_head  import FCOSHead
from train_fcos.utils.box_utils   import batch_nms, clip_boxes_to_image


class FCOSDetector(nn.Module):
    """
    Detector FCOS condicionado por query sketch.
    Solo FPN + FCOSHead + norms del backbone se entrenan.
    """

    def __init__(self, config: dict):
        super().__init__()

        bb_cfg   = config["BACKBONE"]
        fpn_cfg  = config["FPN"]
        head_cfg = config["FCOS_HEAD"]
        qe_cfg   = config["QUERY_ENCODER"]

        # ── Backbone frozen ────────────────────────────────────────────────────
        self.backbone = iDocBackbone(
            arch            = bb_cfg["arch"],
            patch_size      = bb_cfg.get("patch_size", 16),
            embed_dim       = bb_cfg.get("embed_dim", 768),
            depth           = bb_cfg.get("depth", 12),
            num_heads       = bb_cfg.get("num_heads", 12),
            extract_layers  = bb_cfg.get("extract_layers", [2, 5, 8, 11]),
            pretrained_path = config.get("PRETRAINED_PTH"),
        )

        # ── Query encoder (mismo backbone, también frozen) ────────────────────
        self.query_encoder = iDocQueryEncoder(
            backbone  = self.backbone,
            query_dim = bb_cfg.get("embed_dim", 768),
        )

        # ── FPN (entrenable) ──────────────────────────────────────────────────
        self.fpn = FPN(
            in_channels  = fpn_cfg["in_channels"],
            out_channels = fpn_cfg["out_channels"],
        )

        # ── FCOS Head (entrenable) ────────────────────────────────────────────
        self.head = FCOSHead(
            in_channels       = fpn_cfg["out_channels"],
            num_convs         = head_cfg["num_convs"],
            num_classes       = head_cfg["num_classes"],
            query_dim         = qe_cfg["embed_dim"],
            strides           = head_cfg["strides"],
            norm_on_bbox      = head_cfg["norm_on_bbox"],
            centerness_on_reg = head_cfg["centerness_on_reg"],
        )

        self.eval_cfg = config.get("EVAL", {})

    def forward(self, page_imgs: torch.Tensor, query_imgs: torch.Tensor):
        """
        page_imgs:  [B, 3, H, W]  — imágenes de documento a detectar
        query_imgs: [B, 3, Hq, Wq]— sketches de query (misma resolución o diferente)
        Returns:
            (all_cls, all_bbox, all_ctr): salidas raw para la loss
            También disponibles en modo eval: detecciones post-NMS
        """
        # 1. Codificar query sketch → embedding global
        with torch.no_grad():
            query_emb = self.query_encoder(query_imgs)   # [B, 1024]

        # 2. Extraer features multi-escala del backbone (frozen)
        backbone_feats = self.backbone(page_imgs)        # [C2, C3, C4, C5]

        # 3. FPN → [P2, P3, P4, P5, P6]
        fpn_feats = self.fpn(backbone_feats)

        # 4. FCOS head condicionado por query
        all_cls, all_bbox, all_ctr = self.head(fpn_feats, query_emb)

        return all_cls, all_bbox, all_ctr

    @torch.no_grad()
    def predict(self, page_imgs: torch.Tensor, query_imgs: torch.Tensor,
                img_shape: tuple = None) -> list:
        """
        Inferencia completa con NMS.
        Returns: lista de dicts {boxes, scores, labels} por imagen.
        """
        self.eval()
        all_cls, all_bbox, all_ctr = self.forward(page_imgs, query_imgs)

        if img_shape is None:
            img_shape = (page_imgs.shape[-2], page_imgs.shape[-1])

        score_thr = self.eval_cfg.get("score_threshold", 0.05)
        nms_thr   = self.eval_cfg.get("nms_iou_thresh", 0.5)
        max_dets  = self.eval_cfg.get("max_dets", 100)

        raw_dets = self.head.decode_predictions(
            all_cls, all_bbox, all_ctr, img_shape, score_thr
        )

        final_dets = []
        for det in raw_dets:
            if len(det["boxes"]) == 0:
                final_dets.append(det)
                continue

            # NMS por clase
            keep = batch_nms(det["boxes"], det["scores"], det["labels"], nms_thr)
            # Limitar a max_dets por score
            if len(keep) > max_dets:
                top_scores = det["scores"][keep]
                _, sort_idx = top_scores.sort(descending=True)
                keep = keep[sort_idx[:max_dets]]

            final_dets.append({
                "boxes":  clip_boxes_to_image(det["boxes"][keep],  img_shape),
                "scores": det["scores"][keep],
                "labels": det["labels"][keep],
            })

        return final_dets

    def get_trainable_params(self):
        """
        Retorna solo los parámetros entrenables:
        FPN + FCOSHead + LayerNorms del backbone.
        El Swin está frozen.
        """
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(param)
        return trainable