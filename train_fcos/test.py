"""
test.py — Evaluación completa del modelo FCOS-iDoc

Genera:
  - mAP global y por clase (con ranking)
  - Curvas Precision-Recall por clase
  - Análisis de errores: FP y FN por clase
  - Visualización de detecciones sobre imágenes del test set
  - Reporte en consola y guardado en JSON

Uso:
    python -m train_fcos.test \
        --checkpoint /home/rvdl_2/train_fcos/run_01/best_model.pth \
        --image_root /home/rvdl_2/ \
        --dataset_json /home/rvdl_2/detection_dataset_sketches.json \
        --output_dir /home/rvdl_2/train_fcos/test_results/ \
        [--score_thr 0.05] [--iou_thr 0.5] [--vis_n 10]
"""

import os, sys, json, argparse, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")  # sin display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.abspath(os.path.dirname(__file__))
ROOT  = os.path.abspath(os.path.join(_HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_fcos          import config as C
from train_fcos.models.detector   import FCOSDetector
from train_fcos.datasets import build_datasets, collate_fn


# ══════════════════════════════════════════════════════════════════════════════
# 1. ARGUMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Test FCOS iDoc")
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Ruta al checkpoint .pth del modelo.")
    p.add_argument("--image_root",   type=str, required=True,
                   help="Raíz de las imágenes del dataset.")
    p.add_argument("--dataset_json", type=str, default=C.DATASET_JSON)
    p.add_argument("--output_dir",   type=str, default="test_results")
    p.add_argument("--score_thr",    type=float, default=0.05,
                   help="Umbral de score para filtrar detecciones.")
    p.add_argument("--iou_thr",      type=float, default=0.5,
                   help="Umbral IoU para considerar TP.")
    p.add_argument("--nms_iou",      type=float, default=0.5)
    p.add_argument("--vis_n",        type=int, default=10,
                   help="Número de imágenes a visualizar con detecciones.")
    p.add_argument("--split",        type=str, default="test",
                   choices=["test", "val"],
                   help="Evaluar sobre test o val set.")
    p.add_argument("--device",       type=str, default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: torch.device) -> FCOSDetector:
    cfg = {
        "BACKBONE":      C.BACKBONE,
        "FPN":           C.FPN,
        "FCOS_HEAD":     C.FCOS_HEAD,
        "QUERY_ENCODER": C.QUERY_ENCODER,
        "EVAL":          C.EVAL,
        "PRETRAINED_PTH": None,
    }
    model = FCOSDetector(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    epoch   = ckpt.get("epoch", "?")
    best_map = ckpt.get("best_map", None)
    print(f"  Checkpoint: epoch={epoch}"
          + (f"  best_mAP={best_map:.4f}" if best_map else ""))
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def box_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU entre [N,4] y [M,4]. Retorna [N,M]."""
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(0) * (boxes2[:,3]-boxes2[:,1]).clamp(0)
    ix1 = torch.max(boxes1[:,None,0], boxes2[None,:,0])
    iy1 = torch.max(boxes1[:,None,1], boxes2[None,:,1])
    ix2 = torch.min(boxes1[:,None,2], boxes2[None,:,2])
    iy2 = torch.min(boxes1[:,None,3], boxes2[None,:,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    union = area1[:,None] + area2[None,:] - inter
    return inter / union.clamp(1e-6)


class MetricsAccumulator:
    """
    Acumula predicciones y GT para calcular métricas completas al final.
    Guarda también datos para curvas PR y análisis de errores.
    """
    def __init__(self, num_classes: int, iou_thr: float = 0.5):
        self.num_classes = num_classes
        self.iou_thr     = iou_thr
        # Por clase: lista de (score, is_tp)
        self.cls_preds   = defaultdict(list)
        # Por clase: número total de GT
        self.cls_n_gt    = defaultdict(int)
        # Para análisis de errores
        self.fn_per_cls  = defaultdict(int)   # false negatives
        self.fp_per_cls  = defaultdict(int)   # false positives

    def update(self, predictions: list, targets: list):
        """
        predictions: lista de dicts {boxes [N,4], scores [N], labels [N]}
        targets:     lista de dicts {boxes [M,4], labels [M]}
        """
        for pred, tgt in zip(predictions, targets):
            pred_boxes  = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]
            gt_boxes    = tgt["boxes"]
            gt_labels   = tgt["labels"]

            # Contar GT por clase
            for lbl in gt_labels.tolist():
                self.cls_n_gt[lbl] += 1

            # Ordenar predicciones por score
            if len(pred_scores) > 0:
                order = pred_scores.argsort(descending=True)
                pred_boxes  = pred_boxes[order]
                pred_scores = pred_scores[order]
                pred_labels = pred_labels[order]

            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

            for i in range(len(pred_boxes)):
                cls    = pred_labels[i].item()
                score  = pred_scores[i].item()

                gt_cls_mask = (gt_labels == cls)
                if gt_cls_mask.sum() == 0 or len(gt_boxes) == 0:
                    self.cls_preds[cls].append((score, False))
                    self.fp_per_cls[cls] += 1
                    continue

                gt_cls_idx = gt_cls_mask.nonzero(as_tuple=False).squeeze(1)
                iou = box_iou_pairwise(
                    pred_boxes[i:i+1], gt_boxes[gt_cls_idx]
                ).squeeze(0)
                best_iou, best_j = iou.max(0)
                best_gt = gt_cls_idx[best_j.item()].item()

                if best_iou >= self.iou_thr and not matched_gt[best_gt]:
                    self.cls_preds[cls].append((score, True))
                    matched_gt[best_gt] = True
                else:
                    self.cls_preds[cls].append((score, False))
                    self.fp_per_cls[cls] += 1

            # Contar FN (GT no detectados)
            for idx, lbl in enumerate(gt_labels.tolist()):
                if not matched_gt[idx]:
                    self.fn_per_cls[lbl] += 1

    def compute_ap(self, scores_tp: list, n_gt: int):
        """AP con interpolación de 11 puntos."""
        if n_gt == 0 or len(scores_tp) == 0:
            return 0.0, np.array([0.]), np.array([0.])

        scores = np.array([s for s, _ in scores_tp])
        tps    = np.array([int(tp) for _, tp in scores_tp])
        order  = np.argsort(-scores)
        tps    = tps[order]

        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(1 - tps)
        rec    = cum_tp / n_gt
        prec   = cum_tp / (cum_tp + cum_fp + 1e-9)

        # Interpolación 11 puntos
        ap = 0.0
        for thr in np.arange(0.0, 1.1, 0.1):
            p = prec[rec >= thr]
            ap += p.max() if len(p) > 0 else 0.0
        ap /= 11.0

        return ap, rec, prec

    def compute(self):
        per_class = {}
        pr_curves = {}

        for cls in range(self.num_classes):
            n_gt      = self.cls_n_gt.get(cls, 0)
            preds     = self.cls_preds.get(cls, [])
            ap, rec, prec = self.compute_ap(preds, n_gt)

            n_pred = len(preds)
            n_tp   = sum(1 for _, tp in preds if tp)
            n_fp   = self.fp_per_cls.get(cls, 0)
            n_fn   = self.fn_per_cls.get(cls, 0)
            recall_at_end = rec[-1] if len(rec) > 0 else 0.0
            prec_at_end   = prec[-1] if len(prec) > 0 else 0.0

            per_class[cls] = {
                "ap":        round(float(ap), 4),
                "n_gt":      n_gt,
                "n_pred":    n_pred,
                "n_tp":      n_tp,
                "n_fp":      n_fp,
                "n_fn":      n_fn,
                "recall":    round(float(recall_at_end), 4),
                "precision": round(float(prec_at_end), 4),
            }
            pr_curves[cls] = {"recall": rec.tolist(), "precision": prec.tolist()}

        # mAP solo sobre clases que tienen GT
        aps = [v["ap"] for v in per_class.values() if v["n_gt"] > 0]
        mAP = float(np.mean(aps)) if aps else 0.0

        return {"mAP": round(mAP, 4), "per_class": per_class, "pr_curves": pr_curves}


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#8338EC", "#06D6A0", "#FB5607", "#3A86FF", "#FFBE0B",
    "#8D99AE", "#EF233C", "#4CC9F0", "#7209B7", "#560BAD",
    "#480CA8", "#3A0CA3", "#3F37C9", "#4361EE", "#4895EF",
    "#B5179E", "#F72585",
]


def draw_boxes(img: Image.Image, boxes, scores, labels,
               class_names: list, score_thr: float,
               gt_boxes=None, gt_labels=None) -> Image.Image:
    """
    Dibuja predicciones (sólido) y GT (punteado) sobre la imagen.
    """
    img  = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # GT en verde punteado
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box, lbl in zip(gt_boxes.tolist(), gt_labels.tolist()):
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)
            name = class_names[lbl] if lbl < len(class_names) else str(lbl)
            draw.text((x1+2, y1+2), f"GT:{name}", fill="#00FF00")

    # Predicciones en color por clase
    for box, score, lbl in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = COLORS[lbl % len(COLORS)]
        name  = class_names[lbl] if lbl < len(class_names) else str(lbl)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_txt = f"{name}: {score:.2f}"
        draw.rectangle([x1, max(0,y1-14), x1+len(label_txt)*6, y1], fill=color)
        draw.text((x1+2, max(0, y1-13)), label_txt, fill="white")

    return img


def plot_pr_curves(pr_curves: dict, per_class: dict,
                   class_names: list, output_dir: str):
    """Genera un plot con todas las curvas PR en una sola figura."""
    # Solo clases con GT > 0
    valid_cls = [c for c, v in per_class.items() if v["n_gt"] > 0]
    n = len(valid_cls)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()

    for ax_idx, cls in enumerate(sorted(valid_cls)):
        rec  = np.array(pr_curves[cls]["recall"])
        prec = np.array(pr_curves[cls]["precision"])
        ap   = per_class[cls]["ap"]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        color = COLORS[cls % len(COLORS)]

        axes[ax_idx].plot(rec, prec, color=color, linewidth=2)
        axes[ax_idx].fill_between(rec, prec, alpha=0.15, color=color)
        axes[ax_idx].set_xlim(0, 1); axes[ax_idx].set_ylim(0, 1.05)
        axes[ax_idx].set_title(f"{name}\nAP={ap:.3f}", fontsize=9)
        axes[ax_idx].set_xlabel("Recall", fontsize=7)
        axes[ax_idx].set_ylabel("Precision", fontsize=7)
        axes[ax_idx].tick_params(labelsize=6)
        axes[ax_idx].grid(True, alpha=0.3)

    # Ocultar ejes vacíos
    for i in range(len(valid_cls), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Curvas Precision-Recall por clase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → pr_curves.png guardado")


def plot_ap_ranking(per_class: dict, class_names: list, mAP: float,
                    output_dir: str):
    """Bar chart horizontal con AP por clase, ordenado de mayor a menor."""
    valid = {c: v for c, v in per_class.items() if v["n_gt"] > 0}
    sorted_cls = sorted(valid.keys(), key=lambda c: valid[c]["ap"], reverse=True)

    names  = [class_names[c] if c < len(class_names) else str(c) for c in sorted_cls]
    aps    = [valid[c]["ap"] for c in sorted_cls]
    colors = [COLORS[c % len(COLORS)] for c in sorted_cls]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.45)))
    bars = ax.barh(range(len(names)), aps, color=colors, edgecolor="white", height=0.7)

    for bar, ap_val in zip(bars, aps):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{ap_val:.3f}", va="center", fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AP@0.5", fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.axvline(mAP, color="red", linestyle="--", linewidth=1.5, label=f"mAP={mAP:.3f}")
    ax.legend(fontsize=9)
    ax.set_title("Average Precision por clase", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "ap_ranking.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → ap_ranking.png guardado")


def plot_error_analysis(per_class: dict, class_names: list, output_dir: str):
    """Stacked bar: TP, FP, FN por clase."""
    valid = {c: v for c, v in per_class.items() if v["n_gt"] > 0}
    sorted_cls = sorted(valid.keys(), key=lambda c: valid[c]["n_gt"], reverse=True)

    names = [class_names[c] if c < len(class_names) else str(c) for c in sorted_cls]
    tps   = [valid[c]["n_tp"] for c in sorted_cls]
    fps   = [valid[c]["n_fp"] for c in sorted_cls]
    fns   = [valid[c]["n_fn"] for c in sorted_cls]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(names)*0.7), 5))
    ax.bar(x - w, tps, w, label="TP (correcto)", color="#2A9D8F")
    ax.bar(x,     fps, w, label="FP (falso positivo)", color="#E63946")
    ax.bar(x + w, fns, w, label="FN (falso negativo)", color="#E9C46A")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cantidad")
    ax.set_title("Análisis de errores por clase (test set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "error_analysis.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → error_analysis.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORTE EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: dict, class_names: list):
    per = metrics["per_class"]
    mAP = metrics["mAP"]

    # Ordenar por AP descendente
    sorted_cls = sorted(
        [c for c, v in per.items() if v["n_gt"] > 0],
        key=lambda c: per[c]["ap"], reverse=True
    )

    print("\n" + "═"*75)
    print(f"  RESULTADOS TEST SET   mAP@0.5 = {mAP:.4f}")
    print("═"*75)
    print(f"  {'Clase':<16} {'AP':>6}  {'GT':>5}  {'Pred':>5}  "
          f"{'TP':>5}  {'FP':>5}  {'FN':>5}  {'Recall':>7}  {'Prec':>7}")
    print("  " + "-"*71)

    for cls in sorted_cls:
        v    = per[cls]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        bar  = "█" * int(v["ap"] * 20)
        print(f"  {name:<16} {v['ap']:>6.3f}  {v['n_gt']:>5}  {v['n_pred']:>5}  "
              f"{v['n_tp']:>5}  {v['n_fp']:>5}  {v['n_fn']:>5}  "
              f"{v['recall']:>7.3f}  {v['precision']:>7.3f}  {bar}")

    print("  " + "-"*71)

    # Clases sin GT
    no_gt = [c for c, v in per.items() if v["n_gt"] == 0]
    if no_gt:
        no_gt_names = [class_names[c] if c < len(class_names) else str(c)
                       for c in no_gt]
        print(f"\n  Clases sin GT en test set: {', '.join(no_gt_names)}")

    # Top 3 mejores y peores
    if len(sorted_cls) >= 3:
        best  = [class_names[c] for c in sorted_cls[:3]]
        worst = [class_names[c] for c in sorted_cls[-3:]]
        print(f"\n  Mejores  → {', '.join(best)}")
        print(f"  Peores   → {', '.join(worst)}")

    print("═"*75 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  iDoc-FCOS — Test Script")
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Split:      {args.split}")
    print(f"  Score thr:  {args.score_thr}  |  IoU thr: {args.iou_thr}")
    print(f"{'='*60}\n")

    # ── Nombres de clase ──────────────────────────────────────────────────────
    with open(args.dataset_json) as f:
        data = json.load(f)
    sorted_ids  = sorted(c["class_id"] for c in data["classes"])
    id_to_name  = {c["class_id"]: c["class_name"] for c in data["classes"]}
    class_names = [id_to_name[cid] for cid in sorted_ids]
    num_classes = len(class_names)
    print(f"  Clases ({num_classes}): {', '.join(class_names)}\n")

    # ── Modelo ────────────────────────────────────────────────────────────────
    print("Cargando modelo...")
    model = load_model(args.checkpoint, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Cargando dataset...")
    cfg = {
        "BACKBONE": C.BACKBONE, "FPN": C.FPN,
        "FCOS_HEAD": C.FCOS_HEAD, "QUERY_ENCODER": C.QUERY_ENCODER,
        "DATASET": C.DATASET, "AUGMENTATION": C.AUGMENTATION, "EVAL": C.EVAL,
    }
    train_ds, val_ds, test_ds = build_datasets(args.dataset_json, args.image_root, cfg)
    ds = test_ds if args.split == "test" else val_ds
    print(f"  {args.split} set: {len(ds)} muestras\n")

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # ── Inferencia + acumulación de métricas ──────────────────────────────────
    print("Ejecutando inferencia...")
    acc    = MetricsAccumulator(num_classes=num_classes, iou_thr=args.iou_thr)
    vis_saved = 0
    t0     = time.time()

    # Ajustar eval config con los umbrales del test
    C.EVAL["score_threshold"] = args.score_thr
    C.EVAL["nms_iou_thresh"]  = args.nms_iou

    NORMALIZE = T.Normalize(mean=C.DATASET["pixel_mean"], std=C.DATASET["pixel_std"])

    for batch_idx, batch in enumerate(loader):
        pages   = batch["page_imgs"].to(device)
        queries = batch["query_imgs"].to(device)
        targets = batch["targets"]
        shapes  = batch["img_shapes"]

        dets = model.predict(pages, queries, shapes[0])

        cpu_dets = [{"boxes":  d["boxes"].cpu(),
                     "scores": d["scores"].cpu(),
                     "labels": d["labels"].cpu()} for d in dets]
        acc.update(cpu_dets, targets)

        # Visualizar las primeras vis_n imágenes
        if vis_saved < args.vis_n:
            # Reconstruir imagen PIL desde tensor (desnormalizar)
            mean = torch.tensor(C.DATASET["pixel_mean"]).view(3,1,1)
            std  = torch.tensor(C.DATASET["pixel_std"]).view(3,1,1)
            img_t = pages[0].cpu() * std + mean
            img_t = img_t.clamp(0, 1)
            img_pil = TF.to_pil_image(img_t)

            det  = cpu_dets[0]
            tgt  = targets[0]
            img_vis = draw_boxes(
                img_pil,
                det["boxes"], det["scores"], det["labels"],
                class_names, args.score_thr,
                gt_boxes=tgt["boxes"], gt_labels=tgt["labels"]
            )

            sid = batch["sample_ids"][0]
            img_vis.save(str(vis_dir / f"sample_{sid:04d}.jpg"))
            vis_saved += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]", end="\r")

    elapsed = time.time() - t0
    print(f"\n  Inferencia completada en {elapsed:.1f}s "
          f"({elapsed/len(ds)*1000:.1f}ms/imagen)")

    # ── Calcular métricas ─────────────────────────────────────────────────────
    print("\nCalculando métricas...")
    metrics = acc.compute()

    # ── Reporte consola ───────────────────────────────────────────────────────
    print_report(metrics, class_names)

    # ── Guardar JSON ──────────────────────────────────────────────────────────
    # Serializar: añadir nombres de clase al JSON
    metrics_out = {
        "mAP": metrics["mAP"],
        "iou_threshold": args.iou_thr,
        "score_threshold": args.score_thr,
        "split": args.split,
        "per_class": {
            class_names[c] if c < len(class_names) else str(c): v
            for c, v in metrics["per_class"].items()
        }
    }
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print(f"  → metrics.json guardado")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerando gráficos...")
    plot_ap_ranking(metrics["per_class"], class_names, metrics["mAP"], str(out_dir))
    plot_pr_curves(metrics["pr_curves"],  metrics["per_class"], class_names, str(out_dir))
    plot_error_analysis(metrics["per_class"], class_names, str(out_dir))

    print(f"\n✓ Resultados guardados en: {out_dir}/")
    print(f"  metrics.json        ← métricas completas por clase")
    print(f"  ap_ranking.png      ← AP por clase ordenado")
    print(f"  pr_curves.png       ← curvas Precision-Recall")
    print(f"  error_analysis.png  ← TP / FP / FN por clase")
    print(f"  visualizations/     ← {vis_saved} imágenes con detecciones")
    print(f"\n  mAP@{args.iou_thr} = {metrics['mAP']:.4f}")


if __name__ == "__main__":
    main()