"""
train.py — Entrenamiento FCOS sobre backbone iDoc congelado.

Uso (desde la raiz del proyecto, e.g. /home/rvdl_2/):
    python -m train_fcos.train \
        --image_root /home/rvdl_2/ \
        --dataset_json /home/rvdl_2/detection_dataset_sketches.json \
        --pretrained /home/rvdl_2/idoc_pretrained.pth \
        --output_dir /home/rvdl_2/train_fcos/run_01/ \
        --epochs 50 --batch_size 2 --lr 1e-4 --num_workers 4
"""

import os, sys, argparse, time, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
# Agregar ROOT al path permite "import train_fcos.xxx" sin ambigüedad.
# NO agregar train_fcos/ al path: eso causa que "import models" encuentre
# train_fcos/models en vez de algún paquete del venv, rompiendo los imports
# cuando se mezclan imports absolutos y de paquete.
_HERE = os.path.abspath(os.path.dirname(__file__))   # /home/rvdl_2/train_fcos/
ROOT  = os.path.abspath(os.path.join(_HERE, ".."))   # /home/rvdl_2/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Todos los imports usan el prefijo completo train_fcos.
from train_fcos.models.detector        import FCOSDetector
from train_fcos.losses.fcos_loss       import FCOSLoss
from train_fcos.datasets               import build_datasets, collate_fn
from train_fcos.utils.metrics          import DetectionEvaluator
import train_fcos.config               as C

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[train] TensorBoard no disponible, se omite.")


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Entrenamiento FCOS sobre iDoc")
    p.add_argument("--image_root",   type=str, required=True)
    p.add_argument("--output_dir",   type=str, default=C.OUTPUT_DIR)
    p.add_argument("--pretrained",   type=str, default=C.PRETRAINED_PTH)
    p.add_argument("--dataset_json", type=str, default=C.DATASET_JSON)
    p.add_argument("--epochs",       type=int,   default=C.TRAIN["epochs"])
    p.add_argument("--batch_size",   type=int,   default=C.TRAIN["batch_size"])
    p.add_argument("--lr",           type=float, default=C.OPTIMIZER["lr"])
    p.add_argument("--num_workers",  type=int,   default=C.TRAIN["num_workers"])
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=C.TRAIN["seed"])
    return p.parse_args()


# ─── Utilidades ───────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        t = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def save_ckpt(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint guardado: {path}")


def load_ckpt(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scaler    and "scaler"    in ckpt: scaler.load_state_dict(ckpt["scaler"])
    ep = ckpt.get("epoch", 0); bmap = ckpt.get("best_map", 0.0)
    print(f"  ✓ Resumiendo desde {path} (epoch {ep}, best mAP={bmap:.4f})")
    return ep, bmap


# ─── Train epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loss_fn, optimizer, loader, epoch, device,
                    scaler=None, clip_grad=1.0, log_freq=20):
    model.train()
    total, n = 0.0, 0
    for it, batch in enumerate(loader):
        pages   = batch["page_imgs"].to(device, non_blocking=True)
        queries = batch["query_imgs"].to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            cls_, bbox_, ctr_ = model(pages, queries)
            losses = loss_fn(cls_, bbox_, ctr_, targets)
            loss   = losses["loss"]

        if not math.isfinite(loss.item()):
            print(f"  ✗ Loss no finita en iter {it}. Saltando.")
            optimizer.zero_grad(); continue

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total += loss.item(); n += 1
        if it % log_freq == 0:
            print(f"  [{it}/{len(loader)}] "
                  f"loss={loss.item():.4f} cls={losses['loss_cls'].item():.4f} "
                  f"bbox={losses['loss_bbox'].item():.4f} "
                  f"ctr={losses['loss_ctr'].item():.4f} "
                  f"n_pos={losses['n_pos']:.0f}")

    return total / max(n, 1)


# ─── Val epoch ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, num_classes, eval_cfg):
    model.eval()
    ev = DetectionEvaluator(num_classes=num_classes,
                             iou_threshold=eval_cfg["iou_threshold"])
    for batch in loader:
        pages   = batch["page_imgs"].to(device)
        queries = batch["query_imgs"].to(device)
        shapes  = batch["img_shapes"]
        dets = model.predict(pages, queries, shapes[0])
        ev.update(
            [{"boxes": d["boxes"].cpu(), "scores": d["scores"].cpu(),
              "labels": d["labels"].cpu()} for d in dets],
            batch["targets"]
        )
    return ev.compute()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs")) if HAS_TB else None

    cfg = {
        "BACKBONE":      C.BACKBONE,
        "FPN":           C.FPN,
        "FCOS_HEAD":     C.FCOS_HEAD,
        "QUERY_ENCODER": C.QUERY_ENCODER,
        "DATASET":       C.DATASET,
        "AUGMENTATION":  C.AUGMENTATION,
        "EVAL":          C.EVAL,
        "PRETRAINED_PTH": args.pretrained,
    }

    # ── Datasets (split por page_id) ──────────────────────────────────────────
    print("Cargando datasets...")
    train_ds, val_ds, _ = build_datasets(args.dataset_json, args.image_root, cfg)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    # ── Modelo ────────────────────────────────────────────────────────────────
    print("Construyendo modelo...")
    model = FCOSDetector(cfg).to(device)
    trainable = model.get_trainable_params()
    n_train = sum(p.numel() for p in trainable)
    n_froz  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Entrenables: {n_train:,} | Congelados: {n_froz:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    cw = train_ds.class_weights.to(device) if C.LOSS["use_class_weights"] else None
    loss_fn = FCOSLoss(
        num_classes=C.FCOS_HEAD["num_classes"], strides=C.FCOS_HEAD["strides"],
        regress_ranges=C.FCOS_HEAD["regress_ranges"],
        focal_alpha=C.LOSS["focal_alpha"], focal_gamma=C.LOSS["focal_gamma"],
        lambda_cls=C.LOSS["lambda_cls"], lambda_bbox=C.LOSS["lambda_bbox"],
        lambda_ctr=C.LOSS["lambda_ctr"], norm_on_bbox=C.FCOS_HEAD["norm_on_bbox"],
        class_weights=cw,
    ).to(device)

    # ── Optimizador ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=C.OPTIMIZER["weight_decay"],
                                  betas=C.OPTIMIZER["betas"])
    scaler = torch.amp.GradScaler() if C.TRAIN["use_fp16"] else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_map, no_improve = 0, 0.0, 0
    if args.resume:
        start_epoch, best_map = load_ckpt(args.resume, model, optimizer, scaler)

    # ── Loop ──────────────────────────────────────────────────────────────────
    print(f"\nEntrenando {args.epochs} épocas...\n")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs,
                       C.SCHEDULER["warmup_epochs"], args.lr, C.SCHEDULER["min_lr"])
        print(f"{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}  lr={lr:.2e}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(model, loss_fn, optimizer, train_loader,
                                     epoch, device, scaler,
                                     C.TRAIN["clip_grad_norm"], C.TRAIN["log_freq"])

        val_m   = evaluate(model, val_loader, device, C.FCOS_HEAD["num_classes"], C.EVAL)
        val_map = val_m["mAP"]
        print(f"  → loss={train_loss:.4f}  val_mAP={val_map:.4f}")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("mAP/val",    val_map,    epoch)
            writer.add_scalar("LR",         lr,         epoch)
            for cid, ap in val_m["per_class"].items():
                writer.add_scalar(f"AP/{cid}", ap, epoch)

        if (epoch + 1) % C.TRAIN["save_every"] == 0:
            save_ckpt({"epoch": epoch+1, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict() if scaler else None,
                       "best_map": best_map, "metrics": val_m},
                      os.path.join(args.output_dir, f"checkpoint_ep{epoch+1:03d}.pth"))

        if val_map > best_map:
            best_map = val_map; no_improve = 0
            save_ckpt({"epoch": epoch+1, "model": model.state_dict(),
                       "best_map": best_map, "metrics": val_m},
                      os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ★ Nuevo mejor mAP: {best_map:.4f}")
        else:
            no_improve += 1
            print(f"  Sin mejora ({no_improve}/{C.TRAIN['early_stop_patience']})")
            if no_improve >= C.TRAIN["early_stop_patience"]:
                print("Early stopping."); break

    print(f"\nListo en {(time.time()-t0)/3600:.1f}h | Mejor mAP: {best_map:.4f}")
    if writer: writer.close()

if __name__ == "__main__":
    main()