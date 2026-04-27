"""
config.py
Todos los hiperparámetros del entrenamiento FCOS sobre backbone iDoc congelado.
"""

import os

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IDOC_DIR        = os.path.join(ROOT_DIR, "iDoc")
PRETRAINED_PTH  = os.path.join(ROOT_DIR, "idoc_pretrained.pth")
DATASET_JSON    = os.path.join(ROOT_DIR, "detection_dataset_sketches.json")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "train_fcos", "outputs")

# ─── Backbone (ViT-Base, frozen) ──────────────────────────────────────────────
# El checkpoint idoc_pretrained.pth contiene un ViT-Base:
#   patch_embed.proj.weight: [768, 3, 16, 16]  → embed_dim=768, patch_size=16
# Se extraen features de 4 capas intermedias y se escalan a distintos strides
# para simular una pirámide tipo FPN (Simple Feature Pyramid for ViT).
BACKBONE = dict(
    arch         = "vit_base",   # vit_base del iDoc/vision_transformer.py
    patch_size   = 16,
    embed_dim    = 768,
    depth        = 12,
    num_heads    = 12,
    # Índices de capas (0-based) de donde extraer features para cada nivel FPN
    # Capa 2 → P2 (stride 4  via 4x upsample)
    # Capa 5 → P3 (stride 8  via 2x upsample)
    # Capa 8 → P4 (stride 16, resolución nativa del ViT)
    # Capa 11→ P5 (stride 32 via 2x downsample)
    extract_layers = [2, 5, 8, 11],
    # Todos los niveles tienen embed_dim canales antes de entrar al FPN
    out_channels   = [768, 768, 768, 768],
)

# ─── FPN ──────────────────────────────────────────────────────────────────────
FPN = dict(
    in_channels  = BACKBONE["out_channels"],   # [768, 768, 768, 768]
    out_channels = 256,                        # todos los niveles proyectados a 256
    num_levels   = 5,                          # P2-P6
    # P6 se genera por stride-2 max-pool sobre P5
)

# ─── Query encoder (mismo backbone frozen, produce CLS token) ─────────────────
QUERY_ENCODER = dict(
    embed_dim    = BACKBONE["embed_dim"],           # 768 → dim del CLS token ViT
    film_out_dim = FPN["out_channels"],             # FiLM opera sobre canales FPN
)

# ─── FCOS Head ────────────────────────────────────────────────────────────────
FCOS_HEAD = dict(
    in_channels  = FPN["out_channels"],   # 256
    num_convs    = 4,                     # capas conv en cls y reg branches
    num_classes  = 22,                    # 22 clases del dataset HORAE
    # Rangos de tamaño (sqrt area) por nivel FPN — calibrados al dataset
    # marqeur median=27px → P2;  obj_2 median=599px → P6
    regress_ranges = (
        (0,   32),    # P2 stride 4
        (32,  64),    # P3 stride 8
        (64,  128),   # P4 stride 16
        (128, 256),   # P5 stride 32
        (256, 1e8),   # P6 stride 64
    ),
    strides      = [4, 8, 16, 32, 64],
    centerness_on_reg = True,   # rama centerness junto a regresión (más estable)
    norm_on_bbox      = True,   # normalizar targets l,r,t,b por stride
)

# ─── Loss ─────────────────────────────────────────────────────────────────────
LOSS = dict(
    # Focal loss clasificación
    focal_alpha  = 0.25,
    focal_gamma  = 2.0,
    # Peso clase inverso a frecuencia (sqrt para no castigar demasiado)
    # Calculado en runtime desde el dataset
    use_class_weights = True,
    # GIoU para regresión
    loss_bbox_weight  = 1.0,
    # Centerness
    loss_centerness_weight = 1.0,
    # Pesos globales de cada término
    lambda_cls   = 1.0,
    lambda_bbox  = 1.0,
    lambda_ctr   = 1.0,
)

# ─── Dataset & Augmentación ───────────────────────────────────────────────────
DATASET = dict(
    train_ratio  = 0.8,
    val_ratio    = 0.1,
    test_ratio   = 0.1,
    seed         = 42,
    # Resolución de entrada: resize lado corto, tope lado largo
    min_size     = 800,
    max_size     = 1333,
    # Pixel mean/std igual al pre-entrenamiento (ImageNet)
    pixel_mean   = [0.485, 0.456, 0.406],
    pixel_std    = [0.229, 0.224, 0.225],
)

AUGMENTATION = dict(
    # Multi-scale training: lado corto muestreado en este rango
    multi_scale_sizes = [640, 720, 800, 900, 1024],
    # Flip horizontal
    flip_prob         = 0.5,
    # Color jitter (moderado, documentos históricos)
    brightness        = 0.3,
    contrast          = 0.3,
    saturation        = 0.2,
    hue               = 0.05,
    color_jitter_prob = 0.8,
    # Copy-paste: crucial para objetos pequeños (marqeur, croix)
    copy_paste_prob   = 0.5,
    copy_paste_max_objects = 8,
    # Clases prioritarias para copy-paste (las más pequeñas y frecuentes)
    copy_paste_priority_classes = ["marqeur", "croix", "pdp", "S", "T", "petit_A"],
)

# ─── Optimizador ──────────────────────────────────────────────────────────────
OPTIMIZER = dict(
    type          = "AdamW",
    lr            = 1e-4,
    weight_decay  = 1e-4,
    betas         = (0.9, 0.999),
)

SCHEDULER = dict(
    type           = "cosine",
    warmup_epochs  = 5,
    min_lr         = 1e-6,
)

# ─── Entrenamiento ────────────────────────────────────────────────────────────
TRAIN = dict(
    epochs             = 80,
    batch_size         = 4,
    num_workers        = 4,
    clip_grad_norm     = 1.0,
    # Guardar checkpoint cada N épocas
    save_every         = 10,
    # Early stopping: detener si val mAP no mejora en N épocas
    early_stop_patience = 1000,
    # Usar mixed precision
    use_fp16           = True,
    # Frecuencia de log (iteraciones)
    log_freq           = 20,
    # Semilla
    seed               = 42,
)

# ─── Evaluación ───────────────────────────────────────────────────────────────
EVAL = dict(
    iou_threshold    = 0.5,
    score_threshold  = 0.05,
    nms_iou_thresh   = 0.5,
    max_dets         = 100,
)