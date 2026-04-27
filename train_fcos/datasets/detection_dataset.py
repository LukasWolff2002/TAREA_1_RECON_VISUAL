"""
datasets/detection_dataset.py

Dataset de detección cargado desde el JSON de sketches.
Cada muestra contiene:
  - page_img:  imagen de documento completo
  - query_img: sketch de la clase a detectar (uno aleatorio del query pool)
  - boxes:     [M, 4] en xyxy (coordenadas de la imagen escalada)
  - labels:    [M] índices de clase

Estrategia de resolución:
  - Resize lado corto a min_size, tope en max_size
  - Multi-scale training: min_size se muestrea de una lista
"""

import os
import json
import random
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def build_class_index(classes: list) -> dict:
    """Mapea class_id → índice continuo 0..N-1."""
    sorted_ids = sorted(c["class_id"] for c in classes)
    return {cid: idx for idx, cid in enumerate(sorted_ids)}


def build_class_name_index(classes: list) -> dict:
    """Mapea class_name → índice continuo."""
    sorted_ids = sorted(c["class_id"] for c in classes)
    id_to_name = {c["class_id"]: c["class_name"] for c in classes}
    return {id_to_name[cid]: idx for idx, cid in enumerate(sorted_ids)}


def compute_class_weights(samples: list, class_index: dict, num_classes: int) -> torch.Tensor:
    """
    Peso por clase = 1 / sqrt(frecuencia).
    Clases sin boxes reciben peso 1.
    """
    counts = torch.zeros(num_classes)
    for s in samples:
        idx = class_index[s["class_id"]]
        counts[idx] += len(s["boxes"])
    counts = counts.clamp(min=1)
    weights = 1.0 / counts.sqrt()
    weights = weights / weights.mean()   # normalizar para que la media sea 1
    return weights


class HistoricalDocDetectionDataset(Dataset):
    """
    Dataset de detección condicionado por query.

    Cada __getitem__ retorna:
        page_img:  [3, H, W]  tensor normalizado
        query_img: [3, Hq, Wq] tensor normalizado (sketch resizeado a query_size)
        boxes:     [M, 4] float  (xyxy, en coords de page_img escalada)
        labels:    [M]    long
    """

    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        json_path:   str,
        image_root:  str,               # raíz donde están las imágenes
        sample_ids:  list = None,       # subset de sample_id a usar (train/val/test)
        min_size:    int  = 800,
        max_size:    int  = 1333,
        query_size:  int  = 224,        # resolución del query sketch
        augment:     bool = False,
        aug_cfg:     dict = None,
        copy_paste_pool: list = None,   # pool para copy-paste (se setea desde fuera)
    ):
        super().__init__()
        self.image_root = image_root
        self.min_size   = min_size
        self.max_size   = max_size
        self.query_size = query_size
        self.augment    = augment
        self.aug_cfg    = aug_cfg or {}

        # Cargar JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        self.classes    = data["classes"]
        self.class_index     = build_class_index(self.classes)
        self.class_name_idx  = build_class_name_index(self.classes)
        self.num_classes     = len(self.class_index)

        # Query pool: class_id → lista de paths de sketches
        self.query_pool = {}
        for cls in self.classes:
            cid = cls["class_id"]
            self.query_pool[cid] = [q["query_path"] for q in cls.get("queries", [])]

        # Filtrar muestras por sample_id
        all_samples = data["samples"]
        if sample_ids is not None:
            sid_set     = set(sample_ids)
            all_samples = [s for s in all_samples if s["sample_id"] in sid_set]

        self.samples = all_samples

        # Pool externo para copy-paste
        self.copy_paste_pool = copy_paste_pool

        # Normalización
        self.normalize = T.Normalize(mean=self.PIXEL_MEAN, std=self.PIXEL_STD)

        # Compute class weights
        self.class_weights = compute_class_weights(
            all_samples, self.class_index, self.num_classes
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _load_image(self, path: str) -> Image.Image:
        full_path = os.path.join(self.image_root, path)
        img = Image.open(full_path).convert("RGB")
        return img

    def _resize_image_and_boxes(self, img: Image.Image, boxes: np.ndarray,
                                 min_size: int, max_size: int):
        """
        Resize manteniendo aspect ratio.
        Returns: (img_resized, boxes_resized, scale_factor)
        """
        W, H = img.size
        scale = min_size / min(H, W)
        if scale * max(H, W) > max_size:
            scale = max_size / max(H, W)

        new_H, new_W = int(round(H * scale)), int(round(W * scale))
        img = img.resize((new_W, new_H), Image.BILINEAR)

        if len(boxes) > 0:
            boxes = boxes.copy().astype(float)
            boxes[:, [0, 2]] *= (new_W / W)
            boxes[:, [1, 3]] *= (new_H / H)

        return img, boxes, scale

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        x = TF.to_tensor(img)   # [3, H, W] float32 [0,1]
        return self.normalize(x)

    # ─── Augmentación ────────────────────────────────────────────────────────

    def _augment(self, img: Image.Image, boxes: np.ndarray, labels: np.ndarray):
        cfg = self.aug_cfg

        # Multi-scale: samplear min_size de la lista
        multi_sizes = cfg.get("multi_scale_sizes", [self.min_size])
        min_s = random.choice(multi_sizes)
        img, boxes, _ = self._resize_image_and_boxes(img, boxes, min_s, self.max_size)

        # Flip horizontal
        if random.random() < cfg.get("flip_prob", 0.5):
            img = TF.hflip(img)
            if len(boxes) > 0:
                W = img.size[0]
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]

        # Color jitter
        if random.random() < cfg.get("color_jitter_prob", 0.8):
            jitter = T.ColorJitter(
                brightness = cfg.get("brightness", 0.3),
                contrast   = cfg.get("contrast",   0.3),
                saturation = cfg.get("saturation", 0.2),
                hue        = cfg.get("hue",        0.05),
            )
            img = jitter(img)

        # Copy-paste
        if (self.copy_paste_pool is not None and
                random.random() < cfg.get("copy_paste_prob", 0.5)):
            img, boxes, labels = self._copy_paste(img, boxes, labels)

        return img, boxes, labels

    def _copy_paste(self, dst_img: Image.Image, dst_boxes: np.ndarray,
                    dst_labels: np.ndarray):
        """
        Selecciona objetos de imágenes del pool y los pega en dst_img.
        Prioriza las clases configuradas en aug_cfg.
        """
        cfg         = self.aug_cfg
        max_obj     = cfg.get("copy_paste_max_objects", 8)
        priority    = set(cfg.get("copy_paste_priority_classes", []))
        n_to_paste  = random.randint(1, max_obj)

        W_dst, H_dst = dst_img.size
        dst_img      = dst_img.copy()
        new_boxes    = list(dst_boxes)
        new_labels   = list(dst_labels)

        for _ in range(n_to_paste):
            # Elegir muestra del pool (con sesgo hacia clases prioritarias)
            src_sample = self._sample_from_pool(priority)
            if src_sample is None:
                continue

            src_img_pil = self._load_image(src_sample["page_path"])
            if len(src_sample["boxes"]) == 0:
                continue

            # Seleccionar una box aleatoria
            box_info = random.choice(src_sample["boxes"])
            x1, y1, x2, y2 = [int(v) for v in box_info["bbox_xyxy"]]
            x1, x2 = max(0, x1), min(src_img_pil.width,  x2)
            y1, y2 = max(0, y1), min(src_img_pil.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = src_img_pil.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw < 5 or ch < 5:
                continue

            # Pegar en posición aleatoria dentro del destino
            max_px = max(0, W_dst - cw)
            max_py = max(0, H_dst - ch)
            if max_px == 0 or max_py == 0:
                continue
            px = random.randint(0, max_px)
            py = random.randint(0, max_py)

            dst_img.paste(crop, (px, py))
            new_boxes.append([px, py, px + cw, py + ch])
            cls_id  = src_sample["class_id"]
            new_labels.append(self.class_index[cls_id])

        boxes_arr = np.array(new_boxes,  dtype=np.float32) if new_boxes  else dst_boxes
        lbls_arr  = np.array(new_labels, dtype=np.int64)   if new_labels else dst_labels
        return dst_img, boxes_arr, lbls_arr

    def _sample_from_pool(self, priority_names: set):
        """Muestrea una entrada del pool priorizando clases específicas."""
        if not self.copy_paste_pool:
            return None

        # Separar pool en prioritario y resto
        priority_pool = [
            s for s in self.copy_paste_pool
            if s.get("class_name") in priority_names
        ]
        pool = priority_pool if priority_pool and random.random() < 0.7 \
               else self.copy_paste_pool
        return random.choice(pool)

    # ─── __getitem__ ─────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Imagen de documento
        page_img = self._load_image(sample["page_path"])

        # Ground truth boxes
        boxes  = np.array(
            [b["bbox_xyxy"] for b in sample["boxes"]], dtype=np.float32
        ) if sample["boxes"] else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(
            [self.class_index[sample["class_id"]]] * len(sample["boxes"]),
            dtype=np.int64
        )

        # Query: elegir sketch aleatorio del pool de esta clase
        query_paths = self.query_pool.get(sample["class_id"], [])
        if query_paths:
            q_path      = random.choice(query_paths)
            query_img   = self._load_image(q_path)
        else:
            # Fallback: crop de la propia imagen si no hay sketch
            if len(boxes) > 0:
                b = boxes[0].astype(int)
                query_img = page_img.crop((b[0], b[1], b[2], b[3]))
            else:
                query_img = page_img

        # ── Resize ────────────────────────────────────────────────────────────
        if self.augment:
            page_img, boxes, labels = self._augment(page_img, boxes, labels)
        else:
            page_img, boxes, _ = self._resize_image_and_boxes(
                page_img, boxes, self.min_size, self.max_size
            )

        # Query resize fijo a query_size x query_size
        query_img = ImageOps.pad(query_img, (self.query_size, self.query_size))

        # ── A tensor ──────────────────────────────────────────────────────────
        page_tensor  = self._to_tensor(page_img)
        query_tensor = self._to_tensor(query_img)

        boxes_tensor  = torch.from_numpy(boxes).float()
        labels_tensor = torch.from_numpy(labels).long()

        return {
            "page_img":   page_tensor,
            "query_img":  query_tensor,
            "boxes":      boxes_tensor,
            "labels":     labels_tensor,
            "sample_id":  sample["sample_id"],
            "img_shape":  (page_tensor.shape[-2], page_tensor.shape[-1]),
        }

    def __len__(self) -> int:
        return len(self.samples)


def collate_fn(batch: list) -> dict:
    """
    Collate para DataLoader. Las imágenes se pad a la misma resolución en el batch.
    """
    page_imgs   = [b["page_img"]  for b in batch]
    query_imgs  = [b["query_img"] for b in batch]
    boxes       = [b["boxes"]     for b in batch]
    labels      = [b["labels"]    for b in batch]
    sample_ids  = [b["sample_id"] for b in batch]
    img_shapes  = [b["img_shape"] for b in batch]

    # Pad page_imgs al máximo H y W del batch
    max_H = max(img.shape[-2] for img in page_imgs)
    max_W = max(img.shape[-1] for img in page_imgs)

    padded_pages = torch.zeros(len(page_imgs), 3, max_H, max_W)
    for i, img in enumerate(page_imgs):
        padded_pages[i, :, :img.shape[-2], :img.shape[-1]] = img

    query_tensor = torch.stack(query_imgs, dim=0)

    targets = [{"boxes": b, "labels": l} for b, l in zip(boxes, labels)]

    return {
        "page_imgs":  padded_pages,
        "query_imgs": query_tensor,
        "targets":    targets,
        "sample_ids": sample_ids,
        "img_shapes": img_shapes,
    }
