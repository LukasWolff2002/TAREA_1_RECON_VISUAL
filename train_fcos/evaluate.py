"""
evaluate.py

Evaluación de un checkpoint entrenado sobre el test set.
También sirve como script de inferencia sobre una imagen individual.

Uso:
    # Evaluar en test set:
    python train_fcos/evaluate.py \
        --checkpoint train_fcos/outputs/best_model.pth \
        --image_root /ruta/imagenes/

    # Inferencia sobre una imagen + query:
    python train_fcos/evaluate.py \
        --checkpoint train_fcos/outputs/best_model.pth \
        --page_img page.jpg \
        --query_img sketch.jpg \
        --output_img result.jpg
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from pathlib import Path

TRAIN_FCOS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT           = os.path.abspath(os.path.join(TRAIN_FCOS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, TRAIN_FCOS_DIR)

import config as C
from models   import FCOSDetector
from datasets import build_datasets, collate_fn
from utils    import DetectionEvaluator
from torch.utils.data import DataLoader


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser("Evaluación FCOS iDoc")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--image_root",  type=str, default=None)
    parser.add_argument("--dataset_json",type=str, default=C.DATASET_JSON)
    parser.add_argument("--page_img",    type=str, default=None,
                        help="Imagen de página para inferencia single.")
    parser.add_argument("--query_img",   type=str, default=None,
                        help="Sketch de query para inferencia single.")
    parser.add_argument("--output_img",  type=str, default="result.jpg")
    parser.add_argument("--score_thr",   type=float, default=0.3)
    return parser.parse_args()


# ─── Carga del modelo ─────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> FCOSDetector:
    cfg = {
        "BACKBONE":      C.BACKBONE,
        "FPN":           C.FPN,
        "FCOS_HEAD":     C.FCOS_HEAD,
        "QUERY_ENCODER": C.QUERY_ENCODER,
        "EVAL":          C.EVAL,
        "PRETRAINED_PTH": None,   # ya cargamos desde el checkpoint
    }
    model = FCOSDetector(cfg).to(device)

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"Modelo cargado desde {checkpoint_path}")
    if "best_map" in ckpt:
        print(f"  Best mAP reportado: {ckpt['best_map']:.4f}")
    model.eval()
    return model


# ─── Pre-procesado para inferencia single ─────────────────────────────────────

NORMALIZE = T.Normalize(mean=C.DATASET["pixel_mean"], std=C.DATASET["pixel_std"])


def preprocess(img_path: str, min_size: int, max_size: int) -> tuple:
    img = Image.open(img_path).convert("RGB")
    W0, H0 = img.size
    scale  = min_size / min(H0, W0)
    if scale * max(H0, W0) > max_size:
        scale = max_size / max(H0, W0)
    new_H, new_W = int(round(H0 * scale)), int(round(W0 * scale))
    img = img.resize((new_W, new_H), Image.BILINEAR)
    tensor = NORMALIZE(TF.to_tensor(img)).unsqueeze(0)   # [1, 3, H, W]
    return tensor, img, (new_H, new_W)


def preprocess_query(img_path: str, size: int = 224) -> torch.Tensor:
    from PIL import ImageOps
    img    = Image.open(img_path).convert("RGB")
    img    = ImageOps.pad(img, (size, size))
    tensor = NORMALIZE(TF.to_tensor(img)).unsqueeze(0)   # [1, 3, size, size]
    return tensor


# ─── Dibujar detecciones ──────────────────────────────────────────────────────

def draw_detections(img: Image.Image, boxes: torch.Tensor,
                    scores: torch.Tensor, labels: torch.Tensor,
                    class_names: list, score_thr: float = 0.3) -> Image.Image:
    draw   = ImageDraw.Draw(img)
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
              "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#82E0AA"]

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box.tolist()
        cls_name = class_names[label.item()] if label.item() < len(class_names) else str(label.item())
        color    = colors[label.item() % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{cls_name}: {score:.2f}",
                  fill=color)

    return img


# ─── Evaluación en test set ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_set(model: FCOSDetector, image_root: str, device: torch.device):
    cfg = {
        "BACKBONE": C.BACKBONE, "FPN": C.FPN,
        "FCOS_HEAD": C.FCOS_HEAD, "QUERY_ENCODER": C.QUERY_ENCODER,
        "DATASET": C.DATASET, "AUGMENTATION": C.AUGMENTATION,
        "EVAL": C.EVAL,
    }
    _, _, test_ds = build_datasets(C.DATASET_JSON, image_root, cfg)
    print(f"Test set: {len(test_ds)} muestras")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    evaluator = DetectionEvaluator(
        num_classes   = C.FCOS_HEAD["num_classes"],
        iou_threshold = C.EVAL["iou_threshold"],
    )

    for batch in loader:
        page_imgs  = batch["page_imgs"].to(device)
        query_imgs = batch["query_imgs"].to(device)
        targets    = batch["targets"]
        shapes     = batch["img_shapes"]

        dets = model.predict(page_imgs, query_imgs, shapes[0])
        cpu_dets = [{"boxes": d["boxes"].cpu(), "scores": d["scores"].cpu(),
                     "labels": d["labels"].cpu()} for d in dets]
        evaluator.update(cpu_dets, targets)

    metrics = evaluator.compute()
    print(f"\n{'='*50}")
    print(f"  mAP@0.5: {metrics['mAP']:.4f}")
    print(f"{'='*50}")
    print("  AP por clase:")

    # Obtener nombres de clase
    with open(C.DATASET_JSON) as f:
        data = json.load(f)
    sorted_ids = sorted(c["class_id"] for c in data["classes"])
    id_to_name = {c["class_id"]: c["class_name"] for c in data["classes"]}
    class_names = [id_to_name[cid] for cid in sorted_ids]

    for cls_id, ap in sorted(metrics["per_class"].items()):
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        print(f"    {name:15s}: {ap:.4f}")

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)

    # ── Modo inferencia single ────────────────────────────────────────────────
    if args.page_img and args.query_img:
        print(f"\nInferencia: {args.page_img} + query {args.query_img}")

        page_t, page_pil, img_shape = preprocess(
            args.page_img, C.DATASET["min_size"], C.DATASET["max_size"]
        )
        query_t = preprocess_query(args.query_img, C.QUERY_ENCODER.get("size", 224))

        page_t  = page_t.to(device)
        query_t = query_t.to(device)

        dets = model.predict(page_t, query_t, img_shape)
        det  = dets[0]

        # Obtener nombres de clase
        with open(C.DATASET_JSON) as f:
            data = json.load(f)
        sorted_ids  = sorted(c["class_id"] for c in data["classes"])
        id_to_name  = {c["class_id"]: c["class_name"] for c in data["classes"]}
        class_names = [id_to_name[cid] for cid in sorted_ids]

        result = draw_detections(
            page_pil, det["boxes"], det["scores"], det["labels"],
            class_names, args.score_thr
        )
        result.save(args.output_img)
        print(f"  {len(det['boxes'])} detecciones → {args.output_img}")

    # ── Modo evaluación test set ──────────────────────────────────────────────
    elif args.image_root:
        evaluate_test_set(model, args.image_root, device)

    else:
        print("Especifica --image_root para evaluar el test set "
              "o --page_img + --query_img para inferencia single.")


if __name__ == "__main__":
    main()
